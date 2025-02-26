package http

import (
	"context"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/marshaller"
	"github.com/Trendyol/go-triton-client/mocks"
	"github.com/Trendyol/go-triton-client/models"
	"github.com/Trendyol/go-triton-client/options"
	"go.uber.org/mock/gomock"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"
)

func TestNewClient(t *testing.T) {
	_, err := NewClient("http://localhost", false, 1000, 1000, false, false, nil, nil)
	if err == nil {
		t.Errorf("Expected error when URL includes scheme")
	}
	client, err := NewClient("localhost", true, 1000, 1000, true, true, nil, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if client == nil {
		t.Errorf("Expected client to be created")
	}
}

func TestIsServerLive(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
	}
	live, err := c.IsServerLive(context.Background(), &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !live {
		t.Errorf("Expected server to be live")
	}
}

func TestIsServerReady(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
	}
	ready, err := c.IsServerReady(context.Background(), &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !ready {
		t.Errorf("Expected server to be ready")
	}
}

func TestIsModelReady(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
	}
	ready, err := c.IsModelReady(context.Background(), "model", "1", &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !ready {
		t.Errorf("Expected model to be ready")
	}
}

func TestGetServerMetadata(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	metadata := models.ServerMetadataResponse{
		Name: "TestServer",
	}
	bodyBytes, _ := json.Marshal(metadata)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
	}
	response, err := c.GetServerMetadata(context.Background(), &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if response.Name != "TestServer" {
		t.Errorf("Expected server name to be 'TestServer', got '%s'", response.Name)
	}
}

func TestGetModelMetadata(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	metadata := models.ModelMetadataResponse{
		Name: "TestModel",
	}
	bodyBytes, _ := json.Marshal(metadata)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
	}
	response, err := c.GetModelMetadata(context.Background(), "model", "1", &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if response.Name != "TestModel" {
		t.Errorf("Expected model name to be 'TestModel', got '%s'", response.Name)
	}
}

func TestGetModelConfig(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	config := models.ModelConfigResponse{
		Name: "TestModelConfig",
	}
	bodyBytes, _ := json.Marshal(config)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
	}
	response, err := c.GetModelConfig(context.Background(), "model", "1", &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if response.Name != "TestModelConfig" {
		t.Errorf("Expected model config name to be 'TestModelConfig', got '%s'", response.Name)
	}
}

func TestGetModelRepositoryIndex(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	index := []models.ModelRepositoryIndexResponse{
		{
			Name: "Model1",
		},
	}
	bodyBytes, _ := json.Marshal(index)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	mockHttpClient.EXPECT().Post(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
	}
	response, err := c.GetModelRepositoryIndex(context.Background(), &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(response) != 1 || response[0].Name != "Model1" {
		t.Errorf("Expected model repository index to contain 'Model1'")
	}
}

func TestLoadModel(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Post(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     log.Default(),
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.LoadModel(context.Background(), "model", "", nil, &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestUnloadModel(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Post(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     log.Default(),
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UnloadModel(context.Background(), "model", true, &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestGetInferenceStatistics(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	stats := models.InferenceStatisticsResponse{
		ModelStats: []models.InferenceStatisticsModelStat{
			{
				Name: "model",
			},
		},
	}
	bodyBytes, _ := json.Marshal(stats)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
	}
	response, err := c.GetInferenceStatistics(context.Background(), "model", "1", &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(response.ModelStats) != 1 || response.ModelStats[0].Name != "model" {
		t.Errorf("Expected inference statistics for 'model'")
	}
}

func TestGetTraceSettings(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	traceSettings := models.TraceSettingsResponse{
		TraceLevel:   []string{"INFO"},
		TraceRate:    "traceRate",
		TraceCount:   "traceCount",
		LogFrequency: "logFrequency",
		TraceFile:    "traceFile",
	}
	bodyBytes, _ := json.Marshal(traceSettings)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
	}
	response, err := c.GetTraceSettings(context.Background(), "model", &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if response.TraceLevel[0] != "INFO" {
		t.Errorf("Expected trace level to be 'INFO'")
	}
}

func TestUpdateLogSettings(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Post(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     log.Default(),
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UpdateLogSettings(context.Background(), models.LogSettingsRequest{}, &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestGetLogSettings(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	logSettings := models.LogSettingsResponse{
		LogFile:         "local-file.txt",
		LogInfo:         false,
		LogWarning:      false,
		LogError:        false,
		LogVerboseLevel: 1,
		LogFormat:       "%s %s",
	}
	bodyBytes, _ := json.Marshal(logSettings)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
	}
	response, err := c.GetLogSettings(context.Background(), &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if response.LogVerboseLevel != 1 {
		t.Errorf("Expected log setting 'verbose' to be 1")
	}
}

func TestGetSystemSharedMemoryStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	status := []models.SystemSharedMemoryStatusResponse{
		{
			Name: "region1",
		},
	}
	bodyBytes, _ := json.Marshal(status)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
	}
	response, err := c.GetSystemSharedMemoryStatus(context.Background(), "", &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(response) != 1 || response[0].Name != "region1" {
		t.Errorf("Expected system shared memory region 'region1'")
	}
}

func TestRegisterSystemSharedMemory(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Post(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     log.Default(),
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.RegisterSystemSharedMemory(context.Background(), "region1", "key", 1024, 0, &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestUnregisterSystemSharedMemory(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Post(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
	}
	err := c.UnregisterSystemSharedMemory(context.Background(), "region1", &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestGetCUDASharedMemoryStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	status := []models.CUDASharedMemoryStatusResponse{
		{
			Name: "cuda_region1",
		},
	}
	bodyBytes, _ := json.Marshal(status)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
	}
	response, err := c.GetCUDASharedMemoryStatus(context.Background(), "", &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(response) != 1 || response[0].Name != "cuda_region1" {
		t.Errorf("Expected CUDA shared memory region 'cuda_region1'")
	}
}

func TestRegisterCUDASharedMemory(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Post(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
		marshaller: marshaller.NewJSONMarshaller(),
	}
	err := c.RegisterCUDASharedMemory(context.Background(), "cuda_region1", []byte("handle"), 0, 1024, &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestUnregisterCUDASharedMemory(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Post(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
	}
	err := c.UnregisterCUDASharedMemory(context.Background(), "cuda_region1", &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestInfer(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	responseBody := `{
		"model_name": "model",
		"outputs": [
			{
				"name": "output",
				"datatype": "FP32",
				"shape": [1],
				"data": [1.0]
			}
		]
	}`
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(responseBody)),
	}
	mockHttpClient.EXPECT().Do(gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		verbose:    true,
		logger:     log.Default(),
		marshaller: marshaller.NewJSONMarshaller(),
	}
	inputs := []base.InferInput{
		NewInferInput("input", "FP32", []int64{1}, nil),
	}
	outputs := []base.InferOutput{
		NewInferOutput("output", nil),
	}
	result, err := c.Infer(context.Background(), "model", "1", inputs, outputs, &options.InferOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	outputData, err := result.GetOutput("output")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(outputData.GetData()) != 1 || outputData.GetData()[0].(float32) != 1.0 {
		t.Errorf("Expected output data to be [1.0], got %v", outputData.GetData())
	}
}

func TestErrorResponses(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	errorResponse := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       io.NopCloser(strings.NewReader("Internal Server Error")),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).AnyTimes().Return(errorResponse, nil)
	mockHttpClient.EXPECT().Post(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).AnyTimes().Return(errorResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     log.Default(),
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetServerMetadata(context.Background(), &options.Options{})
	if err == nil {
		t.Errorf("Expected error when server returns non-OK status")
	}
	err = c.LoadModel(context.Background(), "model", "", nil, &options.Options{})
	if err == nil {
		t.Errorf("Expected error when server returns non-OK status")
	}
	err = c.UnloadModel(context.Background(), "model", true, &options.Options{})
	if err == nil {
		t.Errorf("Expected error when server returns non-OK status")
	}
}

func TestNewClientDefaultHttpClient(t *testing.T) {
	client, err := NewClient("localhost", true, 1000, 1000, false, false, nil, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if client == nil {
		t.Errorf("Expected client to be created")
	}
}

func TestNewClientWithCustomHttpClient(t *testing.T) {
	customHttpClient := &http.Client{
		Timeout: time.Second,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}
	c, err := NewClient("localhost", true, 1000, 1000, false, true, customHttpClient, log.Default())
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if c == nil {
		t.Errorf("Expected client to be created")
	}
}

func TestInvalidURLInNewClient(t *testing.T) {
	_, err := NewClient("http://localhost", true, 1000, 1000, false, false, nil, log.Default())
	if err == nil {
		t.Errorf("Expected error when URL includes scheme")
	}
}

func TestErrorInHttpClientDo(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockHttpClient.EXPECT().Do(gomock.Any()).Return(nil, errors.New("network error"))
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     log.Default(),
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.Infer(context.Background(), "model", "", nil, nil, &options.InferOptions{})
	if err == nil {
		t.Errorf("Expected error due to network error")
	}
}

func TestNewClient_URLIncludesScheme(t *testing.T) {
	_, err := NewClient("http://localhost", false, 1000, 1000, false, false, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "url should not include the scheme") {
		t.Errorf("Expected error about URL scheme, got %v", err)
	}
}

func TestNewClient_WithNilHttpClientAndLogger(t *testing.T) {
	client, err := NewClient("localhost", false, 1000, 1000, false, false, nil, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if client == nil {
		t.Errorf("Expected client to be created")
	}
}

func TestNewClient_WithCustomHttpClientAndLogger(t *testing.T) {
	customHttpClient := &http.Client{
		Timeout: time.Second,
	}
	customLogger := log.New(io.Discard, "", 0)
	c, err := NewClient("localhost", true, 1000, 1000, true, true, customHttpClient, customLogger)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if c == nil {
		t.Errorf("Expected client to be created")
	}
}

func TestIsServerLive_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2/health/live", gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	live, err := c.IsServerLive(context.Background(), &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !live {
		t.Errorf("Expected server to be live")
	}
}

func TestIsServerLive_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	expectedErr := errors.New("network error")
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2/health/live", gomock.Any(), gomock.Any()).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	live, err := c.IsServerLive(context.Background(), &options.Options{})
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
	if live {
		t.Errorf("Expected live to be false on error")
	}
}

func TestIsServerLive_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusServiceUnavailable,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2/health/live", gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	live, err := c.IsServerLive(context.Background(), &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if live {
		t.Errorf("Expected server to not be live")
	}
}

func TestIsServerReady_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2/health/ready", gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	ready, err := c.IsServerReady(context.Background(), &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !ready {
		t.Errorf("Expected server to be ready")
	}
}

func TestIsServerReady_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	expectedErr := errors.New("network error")
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2/health/ready", gomock.Any(), gomock.Any()).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	ready, err := c.IsServerReady(context.Background(), &options.Options{})
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
	if ready {
		t.Errorf("Expected ready to be false on error")
	}
}

func TestIsServerReady_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusServiceUnavailable,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2/health/ready", gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	ready, err := c.IsServerReady(context.Background(), &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if ready {
		t.Errorf("Expected server to not be ready")
	}
}

func TestIsModelReady_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	modelVersion := "1"
	requestURI := fmt.Sprintf("v2/models/%s/versions/%s/ready", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	ready, err := c.IsModelReady(context.Background(), modelName, modelVersion, &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !ready {
		t.Errorf("Expected model to be ready")
	}
}

func TestIsModelReady_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	modelVersion := "1"
	requestURI := fmt.Sprintf("v2/models/%s/versions/%s/ready", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	expectedErr := errors.New("network error")
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, gomock.Any(), gomock.Any()).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	ready, err := c.IsModelReady(context.Background(), modelName, modelVersion, &options.Options{})
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
	if ready {
		t.Errorf("Expected ready to be false on error")
	}
}

func TestIsModelReady_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	modelVersion := "1"
	requestURI := fmt.Sprintf("v2/models/%s/versions/%s/ready", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	mockResponse := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	ready, err := c.IsModelReady(context.Background(), modelName, modelVersion, &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if ready {
		t.Errorf("Expected model to not be ready")
	}
}

func TestGetServerMetadata_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	expectedErr := errors.New("network error")
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2", gomock.Any(), gomock.Any()).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetServerMetadata(context.Background(), &options.Options{})
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestGetServerMetadata_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2", gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetServerMetadata(context.Background(), &options.Options{})
	if err == nil || !strings.Contains(err.Error(), "failed to get server metadata") {
		t.Errorf("Expected error about failing to get server metadata, got %v", err)
	}
}

func TestGetServerMetadata_InvalidJSON(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	invalidJSON := "{invalid json"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(invalidJSON)),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2", gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetServerMetadata(context.Background(), &options.Options{})
	if err == nil {
		t.Errorf("Expected error due to invalid JSON")
	}
}

func TestGetModelMetadata_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "model"
	modelVersion := "1"
	requestURI := fmt.Sprintf("v2/models/%s/versions/%s", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	expectedErr := errors.New("network error")
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, gomock.Any(), gomock.Any()).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetModelMetadata(context.Background(), modelName, modelVersion, &options.Options{})
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestGetModelMetadata_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "model"
	modelVersion := "1"
	requestURI := fmt.Sprintf("v2/models/%s/versions/%s", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	mockResponse := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetModelMetadata(context.Background(), modelName, modelVersion, &options.Options{})
	if err == nil || !strings.Contains(err.Error(), "failed to get model metadata") {
		t.Errorf("Expected error about failing to get model metadata, got %v", err)
	}
}

func TestGetModelMetadata_InvalidJSON(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "model"
	modelVersion := "1"
	requestURI := fmt.Sprintf("v2/models/%s/versions/%s", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	invalidJSON := "{invalid json"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(invalidJSON)),
	}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, gomock.Any(), gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetModelMetadata(context.Background(), modelName, modelVersion, &options.Options{})
	if err == nil {
		t.Errorf("Expected error due to invalid JSON")
	}
}

func TestLoadModel_InvalidJSON(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "model"
	requestURI := fmt.Sprintf("v2/repository/models/%s/load", url.QueryEscape(modelName))
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, gomock.Any(), gomock.Any(), gomock.Any()).Return(nil, errors.New("network error"))
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     logger,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.LoadModel(context.Background(), modelName, "", nil, &options.Options{})
	if err == nil || !strings.Contains(err.Error(), "network error") {
		t.Errorf("Expected network error, got %v", err)
	}
}

func TestLoadModel_WithConfigAndFiles(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "model"
	requestURI := fmt.Sprintf("v2/repository/models/%s/load", url.QueryEscape(modelName))
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	files := map[string][]byte{
		"file1": []byte("content1"),
	}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, gomock.Any(), gomock.Any(), gomock.Any()).DoAndReturn(
		func(baseURL, uri, body string, headers, params map[string]string) (*http.Response, error) {
			var loadRequest map[string]any
			json.Unmarshal([]byte(body), &loadRequest)
			if (loadRequest["parameters"].(map[string]any)["file1"]).(string) != base64.StdEncoding.EncodeToString([]byte("content1")) {
				t.Errorf("Expected %v, got %v", base64.StdEncoding.EncodeToString([]byte("content1")), (loadRequest["parameters"].(map[string]any)["file1"]).(string))
			}
			return mockResponse, nil
		})
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     logger,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.LoadModel(context.Background(), modelName, "file1", files, &options.Options{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestInfer_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockHttpClient.EXPECT().Do(gomock.Any()).Return(nil, errors.New("network error"))
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
	}
	inputs := []base.InferInput{
		NewInferInput("input", "FP32", []int64{1}, nil),
	}
	_, err := c.Infer(context.Background(), "model", "", inputs, nil, &options.InferOptions{})
	if err == nil || !strings.Contains(err.Error(), "network error") {
		t.Errorf("Expected network error, got %v", err)
	}
}

func TestInfer_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	responseBody := "Internal Server Error"
	mockResponse := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       io.NopCloser(strings.NewReader(responseBody)),
	}
	mockHttpClient.EXPECT().Do(gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
	}
	inputs := []base.InferInput{
		NewInferInput("input", "FP32", []int64{1}, nil),
	}
	_, err := c.Infer(context.Background(), "model", "", inputs, nil, &options.InferOptions{})
	if err == nil || !strings.Contains(err.Error(), "request failed") {
		t.Errorf("Expected error about request failure, got %v", err)
	}
}

func TestInfer_InvalidJSONResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	responseBody := "{invalid json"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(responseBody)),
	}
	mockHttpClient.EXPECT().Do(gomock.Any()).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
	}
	inputs := []base.InferInput{
		NewInferInput("input", "FP32", []int64{1}, nil),
	}
	_, err := c.Infer(context.Background(), "model", "", inputs, nil, &options.InferOptions{})
	if err == nil {
		t.Errorf("Expected error due to invalid JSON response")
	}
}

func TestInfer_PrepareRequestError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockMarshaller := mocks.NewMockMarshaller(ctrl)
	mockMarshaller.EXPECT().Marshal(gomock.Any()).Return(nil, errors.New("marshal error"))

	c := &client{
		baseURL:    "http://localhost",
		marshaller: mockMarshaller,
	}

	inputs := []base.InferInput{
		NewInferInput("input", "FP32", []int64{1}, nil),
	}

	_, err := c.Infer(
		context.Background(),
		"model",
		"",
		inputs,
		nil,
		&options.InferOptions{},
	)
	if err == nil || !strings.Contains(err.Error(), "marshal error") {
		t.Errorf("Expected marshal error, got %v", err)
	}
}

func TestGetModelConfig_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	modelVersion := "1"
	requestURI := fmt.Sprintf("v2/models/%s/versions/%s/config", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	expectedResponse := models.ModelConfigResponse{Name: modelName}
	bodyBytes, _ := json.Marshal(expectedResponse)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	response, err := c.GetModelConfig(context.Background(), modelName, modelVersion, options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if response.Name != modelName {
		t.Errorf("Expected model name '%s', got '%s'", modelName, response.Name)
	}
}

func TestGetModelConfig_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	modelVersion := "1"
	requestURI := fmt.Sprintf("v2/models/%s/versions/%s/config", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	expectedErr := errors.New("network error")
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetModelConfig(context.Background(), modelName, modelVersion, options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestGetModelConfig_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	requestURI := fmt.Sprintf("v2/models/%s/config", url.QueryEscape(modelName))
	mockResponse := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       io.NopCloser(strings.NewReader("Not Found")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetModelConfig(context.Background(), modelName, "", options)
	if err == nil || !strings.Contains(err.Error(), "failed to get model configuration") {
		t.Errorf("Expected error about failing to get model configuration, got %v", err)
	}
}

func TestGetModelConfig_InvalidJSON(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	requestURI := fmt.Sprintf("v2/models/%s/config", url.QueryEscape(modelName))
	invalidJSON := "{invalid json"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(invalidJSON)),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetModelConfig(context.Background(), modelName, "", options)
	if err == nil {
		t.Errorf("Expected error due to invalid JSON")
	}
}

func TestGetModelRepositoryIndex_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	expectedResponse := []models.ModelRepositoryIndexResponse{
		{Name: "model1"},
	}
	bodyBytes, _ := json.Marshal(expectedResponse)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), "v2/repository/index", "", options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	response, err := c.GetModelRepositoryIndex(context.Background(), options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(response) != 1 || response[0].Name != "model1" {
		t.Errorf("Expected response to contain 'model1', got %v", response)
	}
}

func TestGetModelRepositoryIndex_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	expectedErr := errors.New("network error")
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), "v2/repository/index", "", options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetModelRepositoryIndex(context.Background(), options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestGetModelRepositoryIndex_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       io.NopCloser(strings.NewReader("Internal Server Error")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), "v2/repository/index", "", options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetModelRepositoryIndex(context.Background(), options)
	if err == nil || !strings.Contains(err.Error(), "failed to get model repository index") {
		t.Errorf("Expected error about failing to get model repository index, got %v", err)
	}
}

func TestGetModelRepositoryIndex_InvalidJSON(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	invalidJSON := "{invalid json"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(invalidJSON)),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), "v2/repository/index", "", options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetModelRepositoryIndex(context.Background(), options)
	if err == nil {
		t.Errorf("Expected error due to invalid JSON")
	}
}

func TestLoadModel_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	requestURI := fmt.Sprintf("v2/repository/models/%s/load", url.QueryEscape(modelName))
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, gomock.Any(), options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     logger,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.LoadModel(context.Background(), modelName, "", nil, options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestLoadModel_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	requestURI := fmt.Sprintf("v2/repository/models/%s/load", url.QueryEscape(modelName))
	expectedErr := errors.New("network error")
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, gomock.Any(), options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.LoadModel(context.Background(), modelName, "", nil, options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestLoadModel_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	requestURI := fmt.Sprintf("v2/repository/models/%s/load", url.QueryEscape(modelName))
	mockResponse := &http.Response{
		StatusCode: http.StatusBadRequest,
		Body:       io.NopCloser(strings.NewReader("Bad Request")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, gomock.Any(), options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.LoadModel(context.Background(), modelName, "", nil, options)
	if err == nil || !strings.Contains(err.Error(), "failed to load model") {
		t.Errorf("Expected error about failing to load model, got %v", err)
	}
}

func TestUnloadModel_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	requestURI := fmt.Sprintf("v2/repository/models/%s/unload", url.QueryEscape(modelName))
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, gomock.Any(), options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     logger,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UnloadModel(context.Background(), modelName, true, options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestUnloadModel_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	requestURI := fmt.Sprintf("v2/repository/models/%s/unload", url.QueryEscape(modelName))
	expectedErr := errors.New("network error")
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, gomock.Any(), options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UnloadModel(context.Background(), modelName, true, options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestUnloadModel_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	requestURI := fmt.Sprintf("v2/repository/models/%s/unload", url.QueryEscape(modelName))
	mockResponse := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       io.NopCloser(strings.NewReader("Internal Server Error")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, gomock.Any(), options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UnloadModel(context.Background(), modelName, true, options)
	if err == nil || !strings.Contains(err.Error(), "failed to unload model") {
		t.Errorf("Expected error about failing to unload model, got %v", err)
	}
}

func TestGetTraceSettings_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	requestURI := fmt.Sprintf("v2/models/%s/trace/setting", url.QueryEscape(modelName))
	expectedResponse := models.TraceSettingsResponse{
		TraceLevel:   []string{"INFO"},
		TraceRate:    "traceRate",
		TraceCount:   "traceCount",
		LogFrequency: "logFrequency",
		TraceFile:    "dummyFile",
	}
	bodyBytes, _ := json.Marshal(expectedResponse)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     logger,
		verbose:    true,
	}
	response, err := c.GetTraceSettings(context.Background(), modelName, options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if response.TraceLevel[0] != "INFO" {
		t.Errorf("Expected trace level 'INFO', got '%s'", response.TraceLevel[0])
	}
}

func TestGetTraceSettings_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/trace/setting"
	expectedErr := errors.New("network error")
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
		httpClient: mockHttpClient,
	}
	_, err := c.GetTraceSettings(context.Background(), "", options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestGetTraceSettings_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/trace/setting"
	mockResponse := &http.Response{
		StatusCode: http.StatusForbidden,
		Body:       io.NopCloser(strings.NewReader("Forbidden")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetTraceSettings(context.Background(), "", options)
	if err == nil || !strings.Contains(err.Error(), "failed to get trace settings") {
		t.Errorf("Expected error about failing to get trace settings, got %v", err)
	}
}

func TestGetTraceSettings_InvalidJSON(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/trace/setting"
	invalidJSON := "{invalid json"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(invalidJSON)),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
	}
	_, err := c.GetTraceSettings(context.Background(), "", options)
	if err == nil {
		t.Errorf("Expected error due to invalid JSON")
	}
}

func TestUpdateLogSettings_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestBody := models.LogSettingsRequest{
		LogFile:         "logFile",
		LogInfo:         false,
		LogWarning:      false,
		LogError:        false,
		LogVerboseLevel: 1,
		LogFormat:       "%s %s",
	}
	bodyBytes, _ := json.Marshal(requestBody)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), "v2/logging", string(bodyBytes), options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		logger:     logger,
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UpdateLogSettings(context.Background(), requestBody, options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestUpdateLogSettings_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestBody := models.LogSettingsRequest{}
	expectedErr := errors.New("network error")
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), "v2/logging", gomock.Any(), options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UpdateLogSettings(context.Background(), requestBody, options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestUpdateLogSettings_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestBody := models.LogSettingsRequest{}
	mockResponse := &http.Response{
		StatusCode: http.StatusUnauthorized,
		Body:       io.NopCloser(strings.NewReader("Unauthorized")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), "v2/logging", gomock.Any(), options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UpdateLogSettings(context.Background(), requestBody, options)
	if err == nil || !strings.Contains(err.Error(), "failed to update log settings") {
		t.Errorf("Expected error about failing to update log settings, got %v", err)
	}
}

func TestGetLogSettings_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	expectedResponse := models.LogSettingsResponse{
		LogFile:         "logFile",
		LogInfo:         false,
		LogWarning:      false,
		LogError:        false,
		LogVerboseLevel: 1,
		LogFormat:       "%s %s",
	}
	bodyBytes, _ := json.Marshal(expectedResponse)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2/logging", options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		logger:     logger,
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	response, err := c.GetLogSettings(context.Background(), options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if response.LogVerboseLevel != 1 {
		t.Errorf("Expected verbose setting '1', got '%d'", response.LogVerboseLevel)
	}
}

func TestGetLogSettings_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	expectedErr := errors.New("network error")
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2/logging", options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetLogSettings(context.Background(), options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestGetLogSettings_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusForbidden,
		Body:       io.NopCloser(strings.NewReader("Forbidden")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2/logging", options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetLogSettings(context.Background(), options)
	if err == nil || !strings.Contains(err.Error(), "failed to get log settings") {
		t.Errorf("Expected error about failing to get log settings, got %v", err)
	}
}

func TestGetLogSettings_InvalidJSON(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	invalidJSON := "{invalid json"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(invalidJSON)),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), "v2/logging", options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetLogSettings(context.Background(), options)
	if err == nil {
		t.Errorf("Expected error due to invalid JSON")
	}
}

func TestLoadModel_MarshalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockMarshaller := mocks.NewMockMarshaller(ctrl)
	mockMarshaller.EXPECT().Marshal(gomock.Any()).Return(nil, errors.New("marshal error"))

	c := &client{
		baseURL:    "http://localhost",
		marshaller: mockMarshaller,
	}

	err := c.LoadModel(context.Background(), "test_model", "config", nil, &options.Options{})
	if err == nil || !strings.Contains(err.Error(), "marshal error") {
		t.Errorf("Expected marshal error, got %v", err)
	}
}

func TestUnloadModel_MarshalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockMarshaller := mocks.NewMockMarshaller(ctrl)
	mockMarshaller.EXPECT().Marshal(gomock.Any()).Return(nil, errors.New("marshal error"))

	c := &client{
		baseURL:    "http://localhost",
		marshaller: mockMarshaller,
	}

	err := c.UnloadModel(context.Background(), "test_model", true, &options.Options{})
	if err == nil || !strings.Contains(err.Error(), "marshal error") {
		t.Errorf("Expected marshal error, got %v", err)
	}
}

func TestUpdateLogSettings_MarshalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockMarshaller := mocks.NewMockMarshaller(ctrl)
	mockMarshaller.EXPECT().Marshal(gomock.Any()).Return(nil, errors.New("marshal error"))

	c := &client{
		baseURL:    "http://localhost",
		marshaller: mockMarshaller,
	}

	err := c.UpdateLogSettings(context.Background(), models.LogSettingsRequest{}, &options.Options{})
	if err == nil || !strings.Contains(err.Error(), "marshal error") {
		t.Errorf("Expected marshal error, got %v", err)
	}
}

func TestGetSystemSharedMemoryStatus_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	regionName := "region1"
	requestURI := fmt.Sprintf("v2/systemsharedmemory/region/%s/status", url.QueryEscape(regionName))
	expectedResponse := []models.SystemSharedMemoryStatusResponse{
		{Name: regionName},
	}
	bodyBytes, _ := json.Marshal(expectedResponse)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     logger,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	response, err := c.GetSystemSharedMemoryStatus(context.Background(), regionName, options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(response) != 1 || response[0].Name != regionName {
		t.Errorf("Expected region name '%s', got '%v'", regionName, response)
	}
}

func TestGetSystemSharedMemoryStatus_NoRegion(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/systemsharedmemory/status"
	expectedResponse := []models.SystemSharedMemoryStatusResponse{
		{Name: "region1"},
	}
	bodyBytes, _ := json.Marshal(expectedResponse)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
		logger:     log.Default(),
	}
	response, err := c.GetSystemSharedMemoryStatus(context.Background(), "", options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(response) != 1 || response[0].Name != "region1" {
		t.Errorf("Expected region name 'region1', got '%v'", response)
	}
}

func TestGetSystemSharedMemoryStatus_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/systemsharedmemory/status"
	expectedErr := errors.New("network error")
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetSystemSharedMemoryStatus(context.Background(), "", options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestGetSystemSharedMemoryStatus_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/systemsharedmemory/status"
	mockResponse := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       io.NopCloser(strings.NewReader("Internal Server Error")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetSystemSharedMemoryStatus(context.Background(), "", options)
	if err == nil || !strings.Contains(err.Error(), "failed to get system shared memory status") {
		t.Errorf("Expected error about failing to get system shared memory status, got %v", err)
	}
}

func TestGetSystemSharedMemoryStatus_InvalidJSON(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/systemsharedmemory/status"
	invalidJSON := "{invalid json"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(invalidJSON)),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetSystemSharedMemoryStatus(context.Background(), "", options)
	if err == nil {
		t.Errorf("Expected error due to invalid JSON")
	}
}

func TestRegisterSystemSharedMemory_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockMarshaller := mocks.NewMockMarshaller(ctrl)
	mockMarshaller.EXPECT().Marshal(gomock.Any()).Return([]byte(`{"key":"key1","offset":0,"byte_size":1024}`), nil)
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/systemsharedmemory/region/region1/register"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, `{"key":"key1","offset":0,"byte_size":1024}`, options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		marshaller: mockMarshaller,
		httpClient: mockHttpClient,
		logger:     logger,
		verbose:    true,
	}
	err := c.RegisterSystemSharedMemory(context.Background(), "region1", "key1", 1024, 0, options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestRegisterSystemSharedMemory_MarshalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockMarshaller := mocks.NewMockMarshaller(ctrl)
	mockMarshaller.EXPECT().Marshal(gomock.Any()).Return(nil, errors.New("marshal error"))
	c := &client{
		baseURL:    "http://localhost",
		marshaller: mockMarshaller,
	}
	err := c.RegisterSystemSharedMemory(context.Background(), "region1", "key1", 1024, 0, &options.Options{})
	if err == nil || !strings.Contains(err.Error(), "marshal error") {
		t.Errorf("Expected marshal error, got %v", err)
	}
}

func TestRegisterSystemSharedMemory_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockMarshaller := mocks.NewMockMarshaller(ctrl)
	mockMarshaller.EXPECT().Marshal(gomock.Any()).Return([]byte(`{"key":"key1","offset":0,"byte_size":1024}`), nil)
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	expectedErr := errors.New("network error")
	requestURI := "v2/systemsharedmemory/region/region1/register"
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, gomock.Any(), options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		marshaller: mockMarshaller,
		httpClient: mockHttpClient,
	}
	err := c.RegisterSystemSharedMemory(context.Background(), "region1", "key1", 1024, 0, options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestRegisterSystemSharedMemory_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockMarshaller := mocks.NewMockMarshaller(ctrl)
	mockMarshaller.EXPECT().Marshal(gomock.Any()).Return([]byte(`{"key":"key1","offset":0,"byte_size":1024}`), nil)
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusBadRequest,
		Body:       io.NopCloser(strings.NewReader("Bad Request")),
	}
	requestURI := "v2/systemsharedmemory/region/region1/register"
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, gomock.Any(), options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		marshaller: mockMarshaller,
		httpClient: mockHttpClient,
	}
	err := c.RegisterSystemSharedMemory(context.Background(), "region1", "key1", 1024, 0, options)
	if err == nil || !strings.Contains(err.Error(), "failed to register system shared memory") {
		t.Errorf("Expected error about failing to register system shared memory, got %v", err)
	}
}

func TestUnregisterSystemSharedMemory_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/systemsharedmemory/region/region1/unregister"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, "", options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     logger,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UnregisterSystemSharedMemory(context.Background(), "region1", options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestUnregisterSystemSharedMemory_SuccessWithoutName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/systemsharedmemory/unregister"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, "", options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     logger,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UnregisterSystemSharedMemory(context.Background(), "", options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestUnregisterSystemSharedMemory_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	expectedErr := errors.New("network error")
	requestURI := "v2/systemsharedmemory/unregister"
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, "", options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UnregisterSystemSharedMemory(context.Background(), "", options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestUnregisterSystemSharedMemory_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/systemsharedmemory/unregister"
	mockResponse := &http.Response{
		StatusCode: http.StatusForbidden,
		Body:       io.NopCloser(strings.NewReader("Forbidden")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, "", options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UnregisterSystemSharedMemory(context.Background(), "", options)
	if err == nil || !strings.Contains(err.Error(), "failed to unregister system shared memory") {
		t.Errorf("Expected error about failing to unregister system shared memory, got %v", err)
	}
}

func TestGetCUDASharedMemoryStatus_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	regionName := "cuda_region1"
	requestURI := fmt.Sprintf("v2/cudasharedmemory/region/%s/status", url.QueryEscape(regionName))
	expectedResponse := []models.CUDASharedMemoryStatusResponse{
		{Name: regionName},
	}
	bodyBytes, _ := json.Marshal(expectedResponse)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     logger,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	response, err := c.GetCUDASharedMemoryStatus(context.Background(), regionName, options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(response) != 1 || response[0].Name != regionName {
		t.Errorf("Expected CUDA region name '%s', got '%v'", regionName, response)
	}
}

func TestGetCUDASharedMemoryStatus_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/cudasharedmemory/status"
	expectedErr := errors.New("network error")
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetCUDASharedMemoryStatus(context.Background(), "", options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestGetCUDASharedMemoryStatus_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/cudasharedmemory/status"
	mockResponse := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       io.NopCloser(strings.NewReader("Not Found")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetCUDASharedMemoryStatus(context.Background(), "", options)
	if err == nil || !strings.Contains(err.Error(), "failed to get CUDA shared memory status") {
		t.Errorf("Expected error about failing to get CUDA shared memory status, got %v", err)
	}
}

func TestGetCUDASharedMemoryStatus_InvalidJSON(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/cudasharedmemory/status"
	invalidJSON := "{invalid json"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(invalidJSON)),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetCUDASharedMemoryStatus(context.Background(), "", options)
	if err == nil {
		t.Errorf("Expected error due to invalid JSON")
	}
}

func TestRegisterCUDASharedMemory_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockMarshaller := mocks.NewMockMarshaller(ctrl)
	rawHandle := []byte("handle")
	rawHandleBase64 := base64.StdEncoding.EncodeToString(rawHandle)
	expectedRequest := fmt.Sprintf(`{"raw_handle":{"b64":"%s"},"device_id":0,"byte_size":1024}`, rawHandleBase64)
	mockMarshaller.EXPECT().Marshal(gomock.Any()).Return([]byte(expectedRequest), nil)
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/cudasharedmemory/region/cuda_region1/register"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, expectedRequest, options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		marshaller: mockMarshaller,
		httpClient: mockHttpClient,
		logger:     logger,
		verbose:    true,
	}
	err := c.RegisterCUDASharedMemory(context.Background(), "cuda_region1", rawHandle, 0, 1024, options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestRegisterCUDASharedMemory_MarshalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockMarshaller := mocks.NewMockMarshaller(ctrl)
	mockMarshaller.EXPECT().Marshal(gomock.Any()).Return(nil, errors.New("marshal error"))
	c := &client{
		baseURL:    "http://localhost",
		marshaller: mockMarshaller,
	}
	err := c.RegisterCUDASharedMemory(context.Background(), "cuda_region1", []byte("handle"), 0, 1024, &options.Options{})
	if err == nil || !strings.Contains(err.Error(), "marshal error") {
		t.Errorf("Expected marshal error, got %v", err)
	}
}

func TestRegisterCUDASharedMemory_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockMarshaller := mocks.NewMockMarshaller(ctrl)
	rawHandle := []byte("handle")
	rawHandleBase64 := base64.StdEncoding.EncodeToString(rawHandle)
	expectedRequest := fmt.Sprintf(`{"raw_handle":{"b64":"%s"},"device_id":0,"byte_size":1024}`, rawHandleBase64)
	mockMarshaller.EXPECT().Marshal(gomock.Any()).Return([]byte(expectedRequest), nil)
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	expectedErr := errors.New("network error")
	requestURI := "v2/cudasharedmemory/region/cuda_region1/register"
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, expectedRequest, options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		marshaller: mockMarshaller,
		httpClient: mockHttpClient,
	}
	err := c.RegisterCUDASharedMemory(context.Background(), "cuda_region1", rawHandle, 0, 1024, options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestRegisterCUDASharedMemory_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockMarshaller := mocks.NewMockMarshaller(ctrl)
	rawHandle := []byte("handle")
	rawHandleBase64 := base64.StdEncoding.EncodeToString(rawHandle)
	expectedRequest := fmt.Sprintf(`{"raw_handle":{"b64":"%s"},"device_id":0,"byte_size":1024}`, rawHandleBase64)
	mockMarshaller.EXPECT().Marshal(gomock.Any()).Return([]byte(expectedRequest), nil)
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	mockResponse := &http.Response{
		StatusCode: http.StatusForbidden,
		Body:       io.NopCloser(strings.NewReader("Forbidden")),
	}
	requestURI := "v2/cudasharedmemory/region/cuda_region1/register"
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, expectedRequest, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		marshaller: mockMarshaller,
		httpClient: mockHttpClient,
	}
	err := c.RegisterCUDASharedMemory(context.Background(), "cuda_region1", rawHandle, 0, 1024, options)
	if err == nil || !strings.Contains(err.Error(), "failed to register CUDA shared memory") {
		t.Errorf("Expected error about failing to register CUDA shared memory, got %v", err)
	}
}

func TestUnregisterCUDASharedMemory_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/cudasharedmemory/region/cuda_region1/unregister"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, "", options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     logger,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UnregisterCUDASharedMemory(context.Background(), "cuda_region1", options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestUnregisterCUDASharedMemory_SuccessWithoutName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/cudasharedmemory/unregister"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, "", options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     logger,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UnregisterCUDASharedMemory(context.Background(), "", options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestUnregisterCUDASharedMemory_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	expectedErr := errors.New("network error")
	requestURI := "v2/cudasharedmemory/unregister"
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, "", options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UnregisterCUDASharedMemory(context.Background(), "", options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestUnregisterCUDASharedMemory_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	requestURI := "v2/cudasharedmemory/unregister"
	mockResponse := &http.Response{
		StatusCode: http.StatusUnauthorized,
		Body:       io.NopCloser(strings.NewReader("Unauthorized")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Post(gomock.Any(), requestURI, "", options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	err := c.UnregisterCUDASharedMemory(context.Background(), "", options)
	if err == nil || !strings.Contains(err.Error(), "failed to unregister CUDA shared memory") {
		t.Errorf("Expected error about failing to unregister CUDA shared memory, got %v", err)
	}
}

func TestGetInferenceStatistics_Success(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	modelVersion := "1"
	requestURI := fmt.Sprintf("v2/models/%s/versions/%s/stats", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	expectedResponse := models.InferenceStatisticsResponse{
		ModelStats: []models.InferenceStatisticsModelStat{
			{Name: modelName},
		},
	}
	bodyBytes, _ := json.Marshal(expectedResponse)
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(string(bodyBytes))),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	logger := log.Default()
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		logger:     logger,
		verbose:    true,
		marshaller: marshaller.NewJSONMarshaller(),
	}
	response, err := c.GetInferenceStatistics(context.Background(), modelName, modelVersion, options)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(response.ModelStats) != 1 || response.ModelStats[0].Name != modelName {
		t.Errorf("Expected model stats for '%s', got %v", modelName, response.ModelStats)
	}
}

func TestGetInferenceStatistics_NetworkError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	requestURI := fmt.Sprintf("v2/models/%s/stats", url.QueryEscape(modelName))
	expectedErr := errors.New("network error")
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(nil, expectedErr)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetInferenceStatistics(context.Background(), modelName, "", options)
	if err != expectedErr {
		t.Errorf("Expected error '%v', got '%v'", expectedErr, err)
	}
}

func TestGetInferenceStatistics_NonOKStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	requestURI := fmt.Sprintf("v2/models/%s/stats", url.QueryEscape(modelName))
	mockResponse := &http.Response{
		StatusCode: http.StatusServiceUnavailable,
		Body:       io.NopCloser(strings.NewReader("Service Unavailable")),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetInferenceStatistics(context.Background(), modelName, "", options)
	if err == nil || !strings.Contains(err.Error(), "failed to get inference statistics") {
		t.Errorf("Expected error about failing to get inference statistics, got %v", err)
	}
}

func TestGetInferenceStatistics_InvalidJSON(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mockHttpClient := mocks.NewMockHttpClient(ctrl)
	modelName := "test_model"
	requestURI := fmt.Sprintf("v2/models/%s/stats", url.QueryEscape(modelName))
	invalidJSON := "{invalid json"
	mockResponse := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(invalidJSON)),
	}
	options := &options.Options{}
	mockHttpClient.EXPECT().Get(gomock.Any(), requestURI, options.Headers, options.QueryParams).Return(mockResponse, nil)
	c := &client{
		baseURL:    "http://localhost",
		httpClient: mockHttpClient,
		marshaller: marshaller.NewJSONMarshaller(),
		verbose:    true,
	}
	_, err := c.GetInferenceStatistics(context.Background(), modelName, "", options)
	if err == nil {
		t.Errorf("Expected error due to invalid JSON")
	}
}
