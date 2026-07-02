package main

import (
	"bytes"
	"context"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func testConfig(workerURL string) config {
	return config{
		workerURL:      workerURL,
		apiKey:         "secret",
		maxFileBytes:   20 * 1024 * 1024,
		maxConcurrency: 2,
		requestTimeout: 5 * time.Second,
	}
}

func multipartBody(t *testing.T, filename string, content []byte) (*bytes.Buffer, string) {
	t.Helper()
	buf := &bytes.Buffer{}
	mw := multipart.NewWriter(buf)
	part, err := mw.CreateFormFile("file", filename)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(content); err != nil {
		t.Fatal(err)
	}
	_ = mw.Close()
	return buf, mw.FormDataContentType()
}

func TestExtOf(t *testing.T) {
	cases := map[string]string{
		"a.PDF":     "pdf",
		"a.b.jpg":   "jpg",
		"noext":     "",
		"trailing.": "",
		".hidden":   "hidden",
	}
	for in, want := range cases {
		if got := extOf(in); got != want {
			t.Errorf("extOf(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestGetEnvInt(t *testing.T) {
	t.Setenv("TEST_INT", "7")
	if got := getEnvInt("TEST_INT", 3); got != 7 {
		t.Errorf("got %d, want 7", got)
	}
	t.Setenv("TEST_INT", "-1")
	if got := getEnvInt("TEST_INT", 3); got != 3 {
		t.Errorf("negative value should fall back, got %d", got)
	}
	t.Setenv("TEST_INT", "junk")
	if got := getEnvInt("TEST_INT", 3); got != 3 {
		t.Errorf("junk value should fall back, got %d", got)
	}
}

func TestHandleOCRMethodNotAllowed(t *testing.T) {
	s := newServer(testConfig("http://unused"))
	rec := httptest.NewRecorder()
	s.handleOCR(rec, httptest.NewRequest(http.MethodGet, "/ocr", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("got %d, want 405", rec.Code)
	}
}

func TestHandleOCRUnauthorized(t *testing.T) {
	s := newServer(testConfig("http://unused"))
	body, ct := multipartBody(t, "a.pdf", []byte("x"))
	req := httptest.NewRequest(http.MethodPost, "/ocr", body)
	req.Header.Set("Content-Type", ct)
	req.Header.Set("X-API-Key", "wrong")
	rec := httptest.NewRecorder()
	s.handleOCR(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("got %d, want 401", rec.Code)
	}
}

func TestHandleOCRMissingFileField(t *testing.T) {
	s := newServer(testConfig("http://unused"))
	buf := &bytes.Buffer{}
	mw := multipart.NewWriter(buf)
	_ = mw.WriteField("other", "x")
	_ = mw.Close()
	req := httptest.NewRequest(http.MethodPost, "/ocr", buf)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	req.Header.Set("X-API-Key", "secret")
	rec := httptest.NewRecorder()
	s.handleOCR(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("got %d, want 400", rec.Code)
	}
}

func TestHandleOCRBadExtension(t *testing.T) {
	s := newServer(testConfig("http://unused"))
	body, ct := multipartBody(t, "malware.exe", []byte("x"))
	req := httptest.NewRequest(http.MethodPost, "/ocr", body)
	req.Header.Set("Content-Type", ct)
	req.Header.Set("X-API-Key", "secret")
	rec := httptest.NewRecorder()
	s.handleOCR(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("got %d, want 400", rec.Code)
	}
}

func TestHandleOCRProxiesToWorker(t *testing.T) {
	var gotFilename string
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/ocr" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		file, header, err := r.FormFile("file")
		if err != nil {
			t.Errorf("worker got no file: %v", err)
			http.Error(w, "no file", http.StatusBadRequest)
			return
		}
		defer func() { _ = file.Close() }()
		gotFilename = header.Filename
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"text":"hello"}`))
	}))
	defer worker.Close()

	s := newServer(testConfig(worker.URL))
	body, ct := multipartBody(t, "scan.pdf", []byte("%PDF-fake"))
	req := httptest.NewRequest(http.MethodPost, "/ocr", body)
	req.Header.Set("Content-Type", ct)
	req.Header.Set("X-API-Key", "secret")
	rec := httptest.NewRecorder()
	s.handleOCR(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("got %d, want 200, body=%s", rec.Code, rec.Body.String())
	}
	if gotFilename != "scan.pdf" {
		t.Errorf("worker got filename %q, want scan.pdf", gotFilename)
	}
	if !strings.Contains(rec.Body.String(), "hello") {
		t.Errorf("response body not proxied: %s", rec.Body.String())
	}
	if rec.Header().Get("Content-Type") != "application/json" {
		t.Errorf("content-type not proxied: %s", rec.Header().Get("Content-Type"))
	}
}

func TestHandleOCRFilePartTooLarge(t *testing.T) {
	cfg := testConfig("http://unused")
	cfg.maxFileBytes = 100
	s := newServer(cfg)
	body, ct := multipartBody(t, "a.pdf", bytes.Repeat([]byte("x"), 200))
	req := httptest.NewRequest(http.MethodPost, "/ocr", body)
	req.Header.Set("Content-Type", ct)
	req.Header.Set("X-API-Key", "secret")
	rec := httptest.NewRecorder()
	s.handleOCR(rec, req)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("got %d, want 413", rec.Code)
	}
}

func TestHandleOCRWorkerRetrySignalsPassThrough(t *testing.T) {
	for _, code := range []int{http.StatusServiceUnavailable, http.StatusGatewayTimeout} {
		worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			http.Error(w, "busy", code)
		}))
		s := newServer(testConfig(worker.URL))
		body, ct := multipartBody(t, "a.pdf", []byte("x"))
		req := httptest.NewRequest(http.MethodPost, "/ocr", body)
		req.Header.Set("Content-Type", ct)
		req.Header.Set("X-API-Key", "secret")
		rec := httptest.NewRecorder()
		s.handleOCR(rec, req)
		if rec.Code != code {
			t.Errorf("worker %d rewritten to %d, want pass-through", code, rec.Code)
		}
		worker.Close()
	}
}

func TestHandleOCRWorkerErrorBecomes502(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer worker.Close()

	s := newServer(testConfig(worker.URL))
	body, ct := multipartBody(t, "a.pdf", []byte("x"))
	req := httptest.NewRequest(http.MethodPost, "/ocr", body)
	req.Header.Set("Content-Type", ct)
	req.Header.Set("X-API-Key", "secret")
	rec := httptest.NewRecorder()
	s.handleOCR(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Errorf("got %d, want 502", rec.Code)
	}
}

func TestHandleLivez(t *testing.T) {
	s := newServer(testConfig("http://unused"))
	rec := httptest.NewRecorder()
	s.handleLivez(rec, httptest.NewRequest(http.MethodGet, "/livez", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("got %d, want 200", rec.Code)
	}
}

func TestHandleHealthCachesWorkerProbe(t *testing.T) {
	var probes atomic.Int32
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		probes.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer worker.Close()

	s := newServer(testConfig(worker.URL))
	for i := 0; i < 5; i++ {
		rec := httptest.NewRecorder()
		s.handleHealth(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("got %d, want 200", rec.Code)
		}
	}
	if n := probes.Load(); n != 1 {
		t.Errorf("worker probed %d times within TTL, want 1", n)
	}
}

func TestHandleHealthProbeIndependentOfCallerContext(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer worker.Close()

	s := newServer(testConfig(worker.URL))
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // caller already gone when the probe runs
	req := httptest.NewRequest(http.MethodGet, "/health", nil).WithContext(ctx)
	rec := httptest.NewRecorder()
	s.handleHealth(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("cancelled caller context poisoned the shared probe: got %d, want 200", rec.Code)
	}
}

func TestHandleHealthWorkerDown(t *testing.T) {
	s := newServer(testConfig("http://127.0.0.1:1")) // nothing listens here
	rec := httptest.NewRecorder()
	s.handleHealth(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rec.Code != http.StatusBadGateway {
		t.Errorf("got %d, want 502", rec.Code)
	}
}
