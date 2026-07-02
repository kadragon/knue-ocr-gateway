// Command knue-ocr-gateway is the public-facing HTTP entrypoint for the OCR
// service. It validates and forwards uploads to the internal Python OCR
// worker, and does not perform any OCR itself.
package main

import (
	"context"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
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

func loadConfig() config {
	cfg := config{
		workerURL:      getEnv("WORKER_URL", "http://ocr-worker:9000"),
		apiKey:         os.Getenv("API_KEY"),
		maxFileBytes:   int64(getEnvInt("MAX_FILE_MB", 20)) * 1024 * 1024,
		maxConcurrency: getEnvInt("MAX_CONCURRENCY", runtime.NumCPU()),
		requestTimeout: time.Duration(getEnvInt("REQUEST_TIMEOUT_SECONDS", 120)) * time.Second,
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
}

func newServer(cfg config) *server {
	return &server{
		cfg: cfg,
		sem: make(chan struct{}, cfg.maxConcurrency),
		http: &http.Client{
			Timeout: cfg.requestTimeout,
		},
	}
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.cfg.workerURL+"/health", nil)
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	resp, err := s.http.Do(req)
	if err != nil {
		http.Error(w, "worker unavailable", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		http.Error(w, "worker unhealthy", http.StatusBadGateway)
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

	if s.cfg.apiKey != "" && r.Header.Get("X-API-Key") != s.cfg.apiKey {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, s.cfg.maxFileBytes)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		if strings.Contains(err.Error(), "http: request body too large") {
			http.Error(w, "file too large", http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, "invalid multipart form: "+err.Error(), http.StatusBadRequest)
		return
	}

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

	// Bound concurrent forwards to the CPU-bound worker.
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-r.Context().Done():
		return
	}

	pr, pw := io.Pipe()
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

	if resp.StatusCode >= 500 {
		w.WriteHeader(http.StatusBadGateway)
		io.Copy(w, resp.Body)
		return
	}

	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func main() {
	cfg := loadConfig()
	s := newServer(cfg)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/ocr", s.handleOCR)

	addr := ":8080"
	log.Printf("gateway listening on %s, worker=%s, maxConcurrency=%d, maxFileMB=%d",
		addr, cfg.workerURL, cfg.maxConcurrency, cfg.maxFileBytes/(1024*1024))
	log.Fatal(http.ListenAndServe(addr, mux))
}
