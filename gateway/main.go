// Command knue-ocr-gateway is the public-facing HTTP entrypoint for the OCR
// service. It validates and forwards uploads to the internal Python OCR
// worker, and does not perform any OCR itself.
package main

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"flag"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

var allowedExtensions = map[string]bool{
	"pdf":  true,
	"png":  true,
	"jpg":  true,
	"jpeg": true,
	"tiff": true,
	"tif":  true,
	"bmp":  true,
	"webp": true,
}

type config struct {
	workerURL      string
	apiKey         string
	maxFileBytes   int64
	maxConcurrency int
	requestTimeout time.Duration
}

// loadConfig fails startup when no API_KEY is set and the operator hasn't
// explicitly opted into running unauthenticated. This makes "no auth"
// a deliberate choice instead of a silent default for a publicly exposed
// upload endpoint.
func loadConfig() config {
	cfg := config{
		workerURL:      getEnv("WORKER_URL", "http://ocr-worker:9000"),
		apiKey:         os.Getenv("API_KEY"),
		maxFileBytes:   int64(getEnvInt("MAX_FILE_MB", 20)) * 1024 * 1024,
		maxConcurrency: getEnvInt("MAX_CONCURRENCY", runtime.NumCPU()),
		requestTimeout: time.Duration(getEnvInt("REQUEST_TIMEOUT_SECONDS", 120)) * time.Second,
	}
	if cfg.apiKey == "" && os.Getenv("ALLOW_UNAUTHENTICATED") != "true" {
		log.Fatal("API_KEY is not set. Set API_KEY, or set ALLOW_UNAUTHENTICATED=true to run without auth.")
	}
	return cfg
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return fallback
	}
	return n
}

func extOf(filename string) string {
	i := strings.LastIndex(filename, ".")
	if i == -1 || i == len(filename)-1 {
		return ""
	}
	return strings.ToLower(filename[i+1:])
}

type server struct {
	cfg  config
	sem  chan struct{}
	http *http.Client

	// Worker health probe result, cached so the unauthenticated /health
	// endpoint can't be used to hammer the worker with probe traffic.
	healthMu sync.Mutex
	healthOK bool
	healthAt time.Time
}

const healthCacheTTL = 5 * time.Second

func newServer(cfg config) *server {
	return &server{
		cfg: cfg,
		sem: make(chan struct{}, cfg.maxConcurrency),
		http: &http.Client{
			Timeout: cfg.requestTimeout,
		},
	}
}

// handleLivez reports gateway process liveness only, without touching the
// worker: the worker has its own container healthcheck, and coupling the
// gateway's health to it would restart a healthy gateway on worker hiccups.
func (s *server) handleLivez(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"ok"}`))
}

func (s *server) probeWorker(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.cfg.workerURL+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := s.http.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.healthMu.Lock()
	if time.Since(s.healthAt) > healthCacheTTL {
		// Independent of r.Context(): the probe result is cached and shared,
		// so one caller disconnecting mid-probe must not poison the cache
		// with a false negative for everyone else.
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		s.healthOK = s.probeWorker(ctx)
		s.healthAt = time.Now()
		cancel()
	}
	ok := s.healthOK
	s.healthMu.Unlock()

	if !ok {
		http.Error(w, "worker unavailable", http.StatusBadGateway)
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"ok"}`))
}

func (s *server) handleOCR(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if s.cfg.apiKey != "" {
		// Hash both sides first: ConstantTimeCompare short-circuits on length
		// mismatch, which would leak the key's length via timing.
		got := sha256.Sum256([]byte(r.Header.Get("X-API-Key")))
		want := sha256.Sum256([]byte(s.cfg.apiKey))
		if subtle.ConstantTimeCompare(got[:], want[:]) != 1 {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
	}

	// The limit applies to the whole request body; allow 1MB of multipart
	// framing overhead so a file of exactly MAX_FILE_MB is still accepted.
	r.Body = http.MaxBytesReader(w, r.Body, s.cfg.maxFileBytes+1<<20)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		if strings.Contains(err.Error(), "http: request body too large") {
			http.Error(w, "file too large", http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, "invalid multipart form: "+err.Error(), http.StatusBadRequest)
		return
	}
	// ParseMultipartForm spills to disk temp files once the in-memory
	// threshold (32MB) is exceeded; the stdlib does not clean these up.
	defer func() {
		if r.MultipartForm != nil {
			r.MultipartForm.RemoveAll()
		}
	}()

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "missing file field", http.StatusBadRequest)
		return
	}
	defer file.Close()

	ext := extOf(header.Filename)
	if !allowedExtensions[ext] {
		http.Error(w, "unsupported file extension: "+ext, http.StatusBadRequest)
		return
	}

	// MaxBytesReader above only bounds the whole body (with framing
	// headroom); enforce MAX_FILE_MB on the file part itself.
	if header.Size > s.cfg.maxFileBytes {
		http.Error(w, "file too large", http.StatusRequestEntityTooLarge)
		return
	}

	// Bound concurrent forwards to the CPU-bound worker.
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-r.Context().Done():
		return
	}

	pr, pw := io.Pipe()
	// If the request never gets sent (e.g. NewRequestWithContext fails
	// below), nothing reads pr and the writer goroutine below blocks on
	// io.Copy forever. Closing pr unblocks it in that case; it's a no-op
	// once the body has already been fully read by a successful request.
	defer pr.Close()
	mw := multipart.NewWriter(pw)

	go func() {
		defer pw.Close()
		defer mw.Close()
		part, err := mw.CreateFormFile("file", header.Filename)
		if err != nil {
			pw.CloseWithError(err)
			return
		}
		if _, err := io.Copy(part, file); err != nil {
			pw.CloseWithError(err)
			return
		}
	}()

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.requestTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.workerURL+"/ocr", pr)
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := s.http.Do(req)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			http.Error(w, "worker timeout", http.StatusGatewayTimeout)
			return
		}
		http.Error(w, "worker unavailable", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// 503 (worker at capacity) and 504 (processing deadline) are meaningful
	// retry signals from the worker — pass them through. Everything else
	// >=500 is an internal worker failure the client shouldn't see raw.
	if resp.StatusCode >= 500 &&
		resp.StatusCode != http.StatusServiceUnavailable &&
		resp.StatusCode != http.StatusGatewayTimeout {
		w.WriteHeader(http.StatusBadGateway)
		io.Copy(w, resp.Body)
		return
	}

	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

// runHealthcheck probes the local /livez endpoint and exits 0/1. The gateway
// image is distroless (no shell, no curl), so the container healthcheck
// re-invokes this same binary with -healthcheck.
func runHealthcheck() {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get("http://127.0.0.1:8080/livez")
	if err != nil {
		os.Exit(1)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		os.Exit(1)
	}
	os.Exit(0)
}

func main() {
	healthcheck := flag.Bool("healthcheck", false, "probe the local /livez endpoint and exit")
	flag.Parse()
	if *healthcheck {
		runHealthcheck()
	}

	cfg := loadConfig()
	s := newServer(cfg)

	mux := http.NewServeMux()
	mux.HandleFunc("/livez", s.handleLivez)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/ocr", s.handleOCR)

	addr := ":8080"
	log.Printf("gateway listening on %s, worker=%s, maxConcurrency=%d, maxFileMB=%d",
		addr, cfg.workerURL, cfg.maxConcurrency, cfg.maxFileBytes/(1024*1024))

	httpServer := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       cfg.requestTimeout,
		WriteTimeout:      cfg.requestTimeout,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		errCh <- httpServer.ListenAndServe()
	}()

	select {
	case err := <-errCh:
		log.Fatal(err)
	case <-ctx.Done():
	}

	// Let in-flight OCR requests finish; compose must give the container at
	// least this long via stop_grace_period or Docker SIGKILLs us first.
	log.Println("shutting down, draining in-flight requests...")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.requestTimeout+5*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil && !errors.Is(err, context.DeadlineExceeded) {
		log.Printf("shutdown error: %v", err)
	}
	log.Println("gateway stopped")
}
