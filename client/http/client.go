package http

import (
	"context"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/marshaller"
	"github.com/Trendyol/go-triton-client/models"
	"github.com/Trendyol/go-triton-client/options"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type client struct {
	baseURL           string
	verbose           bool
	connectionTimeout float64
	networkTimeout    float64
	ssl               bool
	insecure          bool
	httpClient        base.HttpClient
	marshaller        base.Marshaller
	logger            *log.Logger
}

// NewClient creates a new httpInferenceServerClient.
func NewClient(url string, verbose bool, connectionTimeout float64, networkTimeout float64, ssl bool, insecure bool, httpClient *http.Client, logger *log.Logger) (base.Client, error) {
	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		return nil, fmt.Errorf("url should not include the scheme")
	}

	scheme := "http://"
	if ssl {
		scheme = "https://"
	}

	if httpClient == nil {
		httpClient = &http.Client{
			Timeout: time.Duration(connectionTimeout) * time.Second,
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{InsecureSkipVerify: insecure},
			},
		}
	}

	if logger == nil {
		logger = log.Default()
	}

	return &client{
		baseURL:           scheme + url,
		verbose:           verbose,
		connectionTimeout: connectionTimeout,
		networkTimeout:    networkTimeout,
		ssl:               ssl,
		insecure:          insecure,
		httpClient:        base.NewHttpClient(connectionTimeout, insecure, httpClient),
		logger:            logger,
		marshaller:        marshaller.NewJSONMarshaller(),
	}, nil
}

func (c *client) IsServerLive(ctx context.Context, options *options.Options) (bool, error) {
	resp, err := c.httpClient.Get(c.baseURL, "v2/health/live", options.Headers, options.QueryParams)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK, nil
}

func (c *client) IsServerReady(ctx context.Context, options *options.Options) (bool, error) {
	resp, err := c.httpClient.Get(c.baseURL, "v2/health/ready", options.Headers, options.QueryParams)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK, nil
}

func (c *client) IsModelReady(ctx context.Context, modelName string, modelVersion string, options *options.Options) (bool, error) {
	requestURI := fmt.Sprintf("v2/models/%s/ready", url.QueryEscape(modelName))
	if modelVersion != "" {
		requestURI = fmt.Sprintf("v2/models/%s/versions/%s/ready", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	}

	resp, err := c.httpClient.Get(c.baseURL, requestURI, options.Headers, options.QueryParams)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK, nil
}

func (c *client) GetServerMetadata(ctx context.Context, options *options.Options) (*models.ServerMetadataResponse, error) {
	resp, err := c.httpClient.Get(c.baseURL, "v2", options.Headers, options.QueryParams)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get server metadata. Status code: %d", resp.StatusCode)
	}

	var response models.ServerMetadataResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	if c.verbose {
		c.logger.Println(response)
	}

	return &response, nil
}

func (c *client) GetModelMetadata(ctx context.Context, modelName string, modelVersion string, options *options.Options) (*models.ModelMetadataResponse, error) {
	requestURI := fmt.Sprintf("v2/models/%s", url.QueryEscape(modelName))
	if modelVersion != "" {
		requestURI = fmt.Sprintf("v2/models/%s/versions/%s", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	}

	resp, err := c.httpClient.Get(c.baseURL, requestURI, options.Headers, options.QueryParams)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get model metadata. Status code: %d", resp.StatusCode)
	}

	var response models.ModelMetadataResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	if c.verbose {
		c.logger.Println(response)
	}

	return &response, nil
}

func (c *client) GetModelConfig(ctx context.Context, modelName string, modelVersion string, options *options.Options) (*models.ModelConfigResponse, error) {
	requestURI := fmt.Sprintf("v2/models/%s/config", url.QueryEscape(modelName))
	if modelVersion != "" {
		requestURI = fmt.Sprintf("v2/models/%s/versions/%s/config", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	}

	resp, err := c.httpClient.Get(c.baseURL, requestURI, options.Headers, options.QueryParams)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get model configuration. Status code: %d", resp.StatusCode)
	}

	var response models.ModelConfigResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	if c.verbose {
		c.logger.Println(response)
	}

	return &response, nil
}

func (c *client) GetModelRepositoryIndex(ctx context.Context, options *options.Options) ([]models.ModelRepositoryIndexResponse, error) {
	resp, err := c.httpClient.Post(c.baseURL, "v2/repository/index", "", options.Headers, options.QueryParams)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get model repository index. Status code: %d", resp.StatusCode)
	}

	var response []models.ModelRepositoryIndexResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	if c.verbose {
		c.logger.Println(response)
	}

	return response, nil
}

func (c *client) LoadModel(ctx context.Context, modelName string, config string, files map[string][]byte, options *options.Options) error {
	requestURI := fmt.Sprintf("v2/repository/models/%s/load", url.QueryEscape(modelName))

	loadRequest := make(map[string]any)
	parameters := make(map[string]any)

	if config != "" {
		parameters["config"] = config
	}

	if files != nil {
		for path, content := range files {
			parameters[path] = base64.StdEncoding.EncodeToString(content)
		}
	}

	if len(parameters) > 0 {
		loadRequest["parameters"] = parameters
	}

	requestBody, err := c.marshaller.Marshal(loadRequest)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Post(c.baseURL, requestURI, string(requestBody), options.Headers, options.QueryParams)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to load model '%s'. Status code: %d", modelName, resp.StatusCode)
	}

	if c.verbose {
		c.logger.Printf("Loaded model '%s'\n", modelName)
	}

	return nil
}

func (c *client) UnloadModel(ctx context.Context, modelName string, unloadDependents bool, options *options.Options) error {
	requestURI := fmt.Sprintf("v2/repository/models/%s/unload", url.QueryEscape(modelName))

	unloadRequest := map[string]any{
		"parameters": map[string]any{
			"unload_dependents": unloadDependents,
		},
	}

	requestBody, err := c.marshaller.Marshal(unloadRequest)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Post(c.baseURL, requestURI, string(requestBody), options.Headers, options.QueryParams)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to unload model '%s'. Status code: %d", modelName, resp.StatusCode)
	}

	if c.verbose {
		c.logger.Printf("Unloaded model '%s'\n", modelName)
	}

	return nil
}

func (c *client) GetInferenceStatistics(ctx context.Context, modelName string, modelVersion string, options *options.Options) (*models.InferenceStatisticsResponse, error) {
	requestURI := fmt.Sprintf("v2/models/%s/stats", url.QueryEscape(modelName))
	if modelVersion != "" {
		requestURI = fmt.Sprintf("v2/models/%s/versions/%s/stats", url.QueryEscape(modelName), url.QueryEscape(modelVersion))
	}

	resp, err := c.httpClient.Get(c.baseURL, requestURI, options.Headers, options.QueryParams)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get inference statistics. Status code: %d", resp.StatusCode)
	}

	var response models.InferenceStatisticsResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	if c.verbose {
		c.logger.Println(response)
	}

	return &response, nil
}

func (c *client) GetTraceSettings(ctx context.Context, modelName string, options *options.Options) (*models.TraceSettingsResponse, error) {
	requestURI := "v2/trace/setting"
	if modelName != "" {
		requestURI = fmt.Sprintf("v2/models/%s/trace/setting", url.QueryEscape(modelName))
	}

	resp, err := c.httpClient.Get(c.baseURL, requestURI, options.Headers, options.QueryParams)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get trace settings. Status code: %d", resp.StatusCode)
	}

	var response models.TraceSettingsResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	if c.verbose {
		c.logger.Println(response)
	}

	return &response, nil
}

func (c *client) UpdateLogSettings(ctx context.Context, request models.LogSettingsRequest, options *options.Options) error {
	requestBody, err := c.marshaller.Marshal(request)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Post(c.baseURL, "v2/logging", string(requestBody), options.Headers, options.QueryParams)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to update log settings. Status code: %d", resp.StatusCode)
	}

	if c.verbose {
		c.logger.Println("Updated log settings")
	}

	return nil
}

func (c *client) GetLogSettings(ctx context.Context, options *options.Options) (*models.LogSettingsResponse, error) {
	resp, err := c.httpClient.Get(c.baseURL, "v2/logging", options.Headers, options.QueryParams)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get log settings. Status code: %d", resp.StatusCode)
	}

	var response models.LogSettingsResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	if c.verbose {
		c.logger.Println(response)
	}

	return &response, nil
}

func (c *client) GetSystemSharedMemoryStatus(ctx context.Context, regionName string, options *options.Options) ([]models.SystemSharedMemoryStatusResponse, error) {
	requestURI := "v2/systemsharedmemory/status"
	if regionName != "" {
		requestURI = fmt.Sprintf("v2/systemsharedmemory/region/%s/status", url.QueryEscape(regionName))
	}

	resp, err := c.httpClient.Get(c.baseURL, requestURI, options.Headers, options.QueryParams)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get system shared memory status. Status code: %d", resp.StatusCode)
	}

	var response []models.SystemSharedMemoryStatusResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	if c.verbose {
		c.logger.Println(response)
	}

	return response, nil
}

func (c *client) RegisterSystemSharedMemory(ctx context.Context, name string, key string, byteSize int, offset int, options *options.Options) error {
	requestURI := fmt.Sprintf("v2/systemsharedmemory/region/%s/register", url.QueryEscape(name))

	registerRequest := map[string]any{
		"key":       key,
		"offset":    offset,
		"byte_size": byteSize,
	}
	requestBody, err := c.marshaller.Marshal(registerRequest)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Post(c.baseURL, requestURI, string(requestBody), options.Headers, options.QueryParams)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to register system shared memory. Status code: %d", resp.StatusCode)
	}

	if c.verbose {
		c.logger.Printf("Registered system shared memory with name '%s'\n", name)
	}

	return nil
}

func (c *client) UnregisterSystemSharedMemory(ctx context.Context, name string, options *options.Options) error {
	requestURI := "v2/systemsharedmemory/unregister"
	if name != "" {
		requestURI = fmt.Sprintf("v2/systemsharedmemory/region/%s/unregister", url.QueryEscape(name))
	}

	resp, err := c.httpClient.Post(c.baseURL, requestURI, "", options.Headers, options.QueryParams)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to unregister system shared memory. Status code: %d", resp.StatusCode)
	}

	if c.verbose {
		if name != "" {
			c.logger.Printf("Unregistered system shared memory with name '%s'\n", name)
		} else {
			c.logger.Println("Unregistered all system shared memory regions")
		}
	}

	return nil
}

func (c *client) GetCUDASharedMemoryStatus(ctx context.Context, regionName string, options *options.Options) ([]models.CUDASharedMemoryStatusResponse, error) {
	requestURI := "v2/cudasharedmemory/status"
	if regionName != "" {
		requestURI = fmt.Sprintf("v2/cudasharedmemory/region/%s/status", url.QueryEscape(regionName))
	}

	resp, err := c.httpClient.Get(c.baseURL, requestURI, options.Headers, options.QueryParams)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get CUDA shared memory status. Status code: %d", resp.StatusCode)
	}

	var response []models.CUDASharedMemoryStatusResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	if c.verbose {
		c.logger.Println(response)
	}

	return response, nil
}

func (c *client) RegisterCUDASharedMemory(ctx context.Context, name string, rawHandle []byte, deviceID int, byteSize int, options *options.Options) error {
	requestURI := fmt.Sprintf("v2/cudasharedmemory/region/%s/register", url.QueryEscape(name))

	rawHandleBase64 := base64.StdEncoding.EncodeToString(rawHandle)

	registerRequest := map[string]any{
		"raw_handle": map[string]string{"b64": rawHandleBase64},
		"device_id":  deviceID,
		"byte_size":  byteSize,
	}
	requestBody, err := c.marshaller.Marshal(registerRequest)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Post(c.baseURL, requestURI, string(requestBody), options.Headers, options.QueryParams)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to register CUDA shared memory. Status code: %d", resp.StatusCode)
	}

	if c.verbose {
		c.logger.Printf("Registered CUDA shared memory with name '%s'\n", name)
	}

	return nil
}

func (c *client) UnregisterCUDASharedMemory(ctx context.Context, name string, options *options.Options) error {
	requestURI := "v2/cudasharedmemory/unregister"
	if name != "" {
		requestURI = fmt.Sprintf("v2/cudasharedmemory/region/%s/unregister", url.QueryEscape(name))
	}

	resp, err := c.httpClient.Post(c.baseURL, requestURI, "", options.Headers, options.QueryParams)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to unregister CUDA shared memory. Status code: %d", resp.StatusCode)
	}

	if c.verbose {
		if name != "" {
			c.logger.Printf("Unregistered CUDA shared memory with name '%s'\n", name)
		} else {
			c.logger.Println("Unregistered all CUDA shared memory regions")
		}
	}

	return nil
}

func (c *client) Infer(
	ctx context.Context,
	modelName string,
	modelVersion string,
	inputs []base.InferInput,
	outputs []base.InferOutput,
	options *options.InferOptions,
) (base.InferResult, error) {
	// Prepare the Inference Request
	requestWrapper := NewRequestWrapper(c.baseURL, modelName, modelVersion, inputs, outputs, c.marshaller, options)

	request, err := requestWrapper.PrepareRequest()
	if err != nil {
		return nil, err
	}

	// Make the HTTP call
	resp, err := c.httpClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed: %s", string(bodyBytes))
	}

	// Map the response to the InferResult model
	responseWrapper := NewResponseWrapper(resp)
	return NewInferResult(responseWrapper, c.verbose)
}
