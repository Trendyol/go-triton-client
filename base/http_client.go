package base

import (
	"bytes"
	"crypto/tls"
	"fmt"
	"net/http"
	"time"
)

type HttpClient interface {
	// Get sends a GET request to the specified requestURI with the provided headers and query parameters.
	// Returns the HTTP response and any error encountered.
	Get(baseURL, requestURI string, headers map[string]string, queryParams map[string]string) (*http.Response, error)
	// Post sends a POST request to the specified requestURI with the provided headers, query parameters, and request body.
	// Returns the HTTP response and any error encountered.
	Post(baseURL, requestURI string, requestBody string, headers map[string]string, queryParams map[string]string) (*http.Response, error)
	// PostWithBytes sends a POST request to the specified requestURI with the provided headers, query parameters, and request body as bytes.
	// Returns the HTTP response and any error encountered.
	PostWithBytes(baseURL, requestURI string, requestBody []byte, headers, queryParams map[string]string) (*http.Response, error)
	Do(request *http.Request) (*http.Response, error)

	Close() error
}

type httpClient struct {
	client *http.Client
}

func NewHttpClient(connectionTimeout float64, insecure bool, client *http.Client) HttpClient {
	if client == nil {
		client = &http.Client{
			Timeout: time.Duration(connectionTimeout) * time.Millisecond,
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{InsecureSkipVerify: insecure},
			},
		}
	}
	return &httpClient{client: client}
}

func (h *httpClient) Get(baseURL, requestURI string, headers map[string]string, queryParams map[string]string) (*http.Response, error) {
	req, err := http.NewRequest("GET", fmt.Sprintf("%s/%s", baseURL, requestURI), nil)
	if err != nil {
		return nil, err
	}

	h.addHeaders(req, headers)
	h.addQueryParameters(req, queryParams)

	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func (h *httpClient) Post(baseURL, requestURI string, requestBody string, headers map[string]string, queryParams map[string]string) (*http.Response, error) {
	req, err := http.NewRequest("POST", fmt.Sprintf("%s/%s", baseURL, requestURI), bytes.NewBufferString(requestBody))
	if err != nil {
		return nil, err
	}

	h.addHeaders(req, headers)
	h.addQueryParameters(req, queryParams)

	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func (h *httpClient) PostWithBytes(baseURL, requestURI string, requestBody []byte, headers, queryParams map[string]string) (*http.Response, error) {
	req, err := http.NewRequest("POST", fmt.Sprintf("%s/%s", baseURL, requestURI), bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, err
	}

	h.addHeaders(req, headers)
	h.addQueryParameters(req, queryParams)

	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func (h *httpClient) Do(request *http.Request) (*http.Response, error) {
	return h.client.Do(request)
}

// addHeaders adds the provided headers to the given HTTP request.
func (h *httpClient) addHeaders(req *http.Request, headers map[string]string) {
	for key, value := range headers {
		req.Header.Add(key, value)
	}
}

// addQueryParameters adds the provided query parameters to the given HTTP request.
func (h *httpClient) addQueryParameters(req *http.Request, queryParams map[string]string) {
	q := req.URL.Query()
	for key, value := range queryParams {
		q.Add(key, value)
	}
	req.URL.RawQuery = q.Encode()
}
func (h *httpClient) Close() error {
	h.client.CloseIdleConnections()
	return nil
}
