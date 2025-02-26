package grpc

import (
	"context"
	"encoding/base64"
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/Trendyol/go-triton-client/models"
	"github.com/Trendyol/go-triton-client/options"
	"google.golang.org/grpc"
	"log"
)

type client struct {
	baseURL           string
	verbose           bool
	connectionTimeout float64
	networkTimeout    float64
	ssl               bool
	insecure          bool
	client            grpc_generated_v2.GRPCInferenceServiceClient
	logger            *log.Logger
}

// NewClient creates a new gRPCInferenceServerClient.
func NewClient(url string, verbose bool, connectionTimeout float64, networkTimeout float64, ssl bool, insecureConnection bool, grpcConnection *grpc.ClientConn, logger *log.Logger) (base.Client, error) {
	if logger == nil {
		logger = log.Default()
	}

	if grpcConnection == nil {
		grpcClient, err := base.NewGrpcClient(url, connectionTimeout, networkTimeout, ssl, insecureConnection)
		if err != nil {
			return nil, err
		}
		grpcConnection = grpcClient.GetConnection()
	}

	return &client{
		baseURL:           url,
		verbose:           verbose,
		connectionTimeout: connectionTimeout,
		networkTimeout:    networkTimeout,
		ssl:               ssl,
		insecure:          insecureConnection,
		client:            grpc_generated_v2.NewGRPCInferenceServiceClient(grpcConnection),
		logger:            logger,
	}, nil
}

func (c *client) IsServerLive(ctx context.Context, options *options.Options) (bool, error) {
	resp, err := c.client.ServerLive(ctx, &grpc_generated_v2.ServerLiveRequest{})
	if err != nil {
		return false, err
	}
	return resp.Live, nil
}

func (c *client) IsServerReady(ctx context.Context, options *options.Options) (bool, error) {
	resp, err := c.client.ServerReady(ctx, &grpc_generated_v2.ServerReadyRequest{})
	if err != nil {
		return false, err
	}
	return resp.Ready, nil
}

func (c *client) IsModelReady(ctx context.Context, modelName, modelVersion string, options *options.Options) (bool, error) {
	resp, err := c.client.ModelReady(ctx, &grpc_generated_v2.ModelReadyRequest{
		Name:    modelName,
		Version: modelVersion,
	})
	if err != nil {
		return false, err
	}
	return resp.Ready, nil
}

func (c *client) GetServerMetadata(ctx context.Context, options *options.Options) (*models.ServerMetadataResponse, error) {
	resp, err := c.client.ServerMetadata(ctx, &grpc_generated_v2.ServerMetadataRequest{})
	if err != nil {
		return nil, err
	}

	response := &models.ServerMetadataResponse{
		Name:       resp.Name,
		Version:    resp.Version,
		Extensions: resp.Extensions,
	}

	if c.verbose {
		c.logger.Println(response)
	}

	return response, nil
}

func (c *client) GetModelMetadata(ctx context.Context, modelName, modelVersion string, options *options.Options) (*models.ModelMetadataResponse, error) {
	req := &grpc_generated_v2.ModelMetadataRequest{
		Name:    modelName,
		Version: modelVersion,
	}

	resp, err := c.client.ModelMetadata(ctx, req)
	if err != nil {
		return nil, err
	}

	if c.verbose {
		c.logger.Println(resp)
	}

	modelMetadata := &models.ModelMetadataResponse{
		Name:     resp.Name,
		Versions: resp.Versions,
		Platform: resp.Platform,
	}

	for _, input := range resp.Inputs {
		inputShape := make([]int, len(input.Shape))
		for i, s := range input.Shape {
			inputShape[i] = int(s)
		}
		modelMetadata.Inputs = append(modelMetadata.Inputs, models.ModelMetadataInput{
			Name:     input.Name,
			Datatype: input.Datatype,
			Shape:    inputShape,
		})
	}

	for _, output := range resp.Outputs {
		outputShape := make([]int, len(output.Shape))
		for i, s := range output.Shape {
			outputShape[i] = int(s)
		}
		modelMetadata.Outputs = append(modelMetadata.Outputs, models.ModelMetadataOutput{
			Name:     output.Name,
			Datatype: output.Datatype,
			Shape:    outputShape,
		})
	}

	if c.verbose {
		c.logger.Println(modelMetadata)
	}

	return modelMetadata, nil
}

func (c *client) GetModelConfig(ctx context.Context, modelName, modelVersion string, options *options.Options) (*models.ModelConfigResponse, error) {
	req := &grpc_generated_v2.ModelConfigRequest{
		Name:    modelName,
		Version: modelVersion,
	}

	resp, err := c.client.ModelConfig(ctx, req)
	if err != nil {
		return nil, err
	}

	configResponse := &models.ModelConfigResponse{
		Name:                 resp.Config.Name,
		Platform:             resp.Config.Platform,
		Backend:              resp.Config.Backend,
		DefaultModelFileName: resp.Config.DefaultModelFilename,
	}

	if resp.Config.VersionPolicy != nil && resp.Config.VersionPolicy.GetLatest() != nil {
		configResponse.VersionPolicy = models.ModelConfigVersionPolicy{
			Latest: models.ModelConfigLatestVersionPolicy{
				NumVersions: int(resp.Config.VersionPolicy.GetLatest().NumVersions),
			},
		}
	}

	for _, input := range resp.Config.Input {
		inputDims := make([]int, len(input.Dims))
		for i, dim := range input.Dims {
			inputDims[i] = int(dim)
		}
		configResponse.Input = append(configResponse.Input, models.ModelConfigInput{
			Name:             input.Name,
			DataType:         input.DataType.String(),
			Format:           input.Format.String(),
			Dims:             inputDims,
			IsShapeTensor:    input.IsShapeTensor,
			AllowRaggedBatch: input.AllowRaggedBatch,
			Optional:         input.Optional,
		})
	}

	for _, output := range resp.Config.Output {
		outputDims := make([]int, len(output.Dims))
		for i, dim := range output.Dims {
			outputDims[i] = int(dim)
		}
		configResponse.Output = append(configResponse.Output, models.ModelConfigOutput{
			Name:          output.Name,
			DataType:      output.DataType.String(),
			Dims:          outputDims,
			LabelFilename: output.LabelFilename,
			IsShapeTensor: output.IsShapeTensor,
		})
	}

	for _, group := range resp.Config.InstanceGroup {
		gpus := make([]string, len(group.Gpus))
		for i, gpu := range group.Gpus {
			gpus[i] = fmt.Sprintf("%d", gpu)
		}
		configResponse.InstanceGroup = append(configResponse.InstanceGroup, models.ModelConfigInstanceGroup{
			Name:       group.Name,
			Kind:       group.Kind.String(),
			Count:      int(group.Count),
			GPUs:       gpus,
			Passive:    group.Passive,
			HostPolicy: group.HostPolicy,
		})
	}

	if c.verbose {
		c.logger.Println(configResponse)
	}

	return configResponse, nil
}

func (c *client) GetModelRepositoryIndex(ctx context.Context, options *options.Options) ([]models.ModelRepositoryIndexResponse, error) {
	req := &grpc_generated_v2.RepositoryIndexRequest{}

	resp, err := c.client.RepositoryIndex(ctx, req)
	if err != nil {
		return nil, err
	}

	var index []models.ModelRepositoryIndexResponse
	for _, model := range resp.Models {
		index = append(index, models.ModelRepositoryIndexResponse{
			Name:    model.Name,
			Version: model.Version,
			State:   model.State,
			Reason:  model.Reason,
		})
	}

	if c.verbose {
		c.logger.Println(index)
	}

	return index, nil
}

func (c *client) LoadModel(ctx context.Context, modelName string, config string, files map[string][]byte, options *options.Options) error {
	loadRequest := &grpc_generated_v2.RepositoryModelLoadRequest{
		ModelName:  modelName,
		Parameters: make(map[string]*grpc_generated_v2.ModelRepositoryParameter),
	}

	if config != "" {
		loadRequest.Parameters["config"] = &grpc_generated_v2.ModelRepositoryParameter{
			ParameterChoice: &grpc_generated_v2.ModelRepositoryParameter_StringParam{
				StringParam: config,
			},
		}
	}

	for path, content := range files {
		loadRequest.Parameters[path] = &grpc_generated_v2.ModelRepositoryParameter{
			ParameterChoice: &grpc_generated_v2.ModelRepositoryParameter_StringParam{
				StringParam: base64.StdEncoding.EncodeToString(content),
			},
		}
	}

	_, err := c.client.RepositoryModelLoad(ctx, loadRequest)
	if err != nil {
		return fmt.Errorf("failed to load model '%s': %w", modelName, err)
	}

	if c.verbose {
		c.logger.Printf("successfully loaded model '%s'\n", modelName)
	}

	return nil
}

func (c *client) UnloadModel(ctx context.Context, modelName string, unloadDependents bool, options *options.Options) error {
	unloadRequest := &grpc_generated_v2.RepositoryModelUnloadRequest{
		ModelName:  modelName,
		Parameters: make(map[string]*grpc_generated_v2.ModelRepositoryParameter),
	}

	unloadRequest.Parameters["unload_dependents"] = &grpc_generated_v2.ModelRepositoryParameter{
		ParameterChoice: &grpc_generated_v2.ModelRepositoryParameter_BoolParam{
			BoolParam: unloadDependents,
		},
	}

	_, err := c.client.RepositoryModelUnload(ctx, unloadRequest)
	if err != nil {
		return fmt.Errorf("failed to unload model '%s': %w", modelName, err)
	}

	if c.verbose {
		c.logger.Printf("successfully unloaded model '%s'\n", modelName)
	}

	return nil
}

func (c *client) GetInferenceStatistics(ctx context.Context, modelName, modelVersion string, options *options.Options) (*models.InferenceStatisticsResponse, error) {
	req := &grpc_generated_v2.ModelStatisticsRequest{
		Name:    modelName,
		Version: modelVersion,
	}

	resp, err := c.client.ModelStatistics(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get inference statistics for model '%s' version '%s': %w", modelName, modelVersion, err)
	}

	inferenceStatsResponse := &models.InferenceStatisticsResponse{
		ModelStats: make([]models.InferenceStatisticsModelStat, len(resp.ModelStats)),
	}

	for i, modelStat := range resp.ModelStats {
		stat := models.InferenceStatisticsModelStat{
			Name:           modelStat.Name,
			Version:        modelStat.Version,
			LastInference:  int(modelStat.LastInference),
			InferenceCount: int(modelStat.InferenceCount),
			ExecutionCount: int(modelStat.ExecutionCount),
			InferenceStats: models.InferenceStatisticsInferenceStats{
				Success:       mapInferenceStat(modelStat.InferenceStats.Success),
				Fail:          mapInferenceStat(modelStat.InferenceStats.Fail),
				Queue:         mapInferenceStat(modelStat.InferenceStats.Queue),
				ComputeInput:  mapInferenceStat(modelStat.InferenceStats.ComputeInput),
				ComputeInfer:  mapInferenceStat(modelStat.InferenceStats.ComputeInfer),
				ComputeOutput: mapInferenceStat(modelStat.InferenceStats.ComputeOutput),
				CacheHit:      mapInferenceStat(modelStat.InferenceStats.CacheHit),
				CacheMiss:     mapInferenceStat(modelStat.InferenceStats.CacheMiss),
			},
			BatchStats:  mapBatchStats(modelStat.BatchStats),
			MemoryUsage: mapMemoryUsage(modelStat.MemoryUsage),
		}

		inferenceStatsResponse.ModelStats[i] = stat
	}

	if c.verbose {
		c.logger.Println(inferenceStatsResponse)
	}

	return inferenceStatsResponse, nil
}

func (c *client) GetTraceSettings(ctx context.Context, modelName string, options *options.Options) (*models.TraceSettingsResponse, error) {
	req := &grpc_generated_v2.TraceSettingRequest{
		ModelName: modelName,
	}

	resp, err := c.client.TraceSetting(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get trace settings for model '%s': %w", modelName, err)
	}

	traceSettings := &models.TraceSettingsResponse{}

	for key, settingValue := range resp.Settings {
		switch key {
		case "trace_level":
			traceSettings.TraceLevel = settingValue.Value
		case "trace_rate":
			if len(settingValue.Value) > 0 {
				traceSettings.TraceRate = settingValue.Value[0]
			}
		case "trace_count":
			if len(settingValue.Value) > 0 {
				traceSettings.TraceCount = settingValue.Value[0]
			}
		case "log_frequency":
			if len(settingValue.Value) > 0 {
				traceSettings.LogFrequency = settingValue.Value[0]
			}
		case "trace_file":
			if len(settingValue.Value) > 0 {
				traceSettings.TraceFile = settingValue.Value[0]
			}
		}
	}

	if c.verbose {
		c.logger.Println(traceSettings)
	}

	return traceSettings, nil
}

func (c *client) UpdateLogSettings(ctx context.Context, request models.LogSettingsRequest, options *options.Options) error {
	logSettingsRequest := &grpc_generated_v2.LogSettingsRequest{
		Settings: make(map[string]*grpc_generated_v2.LogSettingsRequest_SettingValue),
	}

	logSettingsRequest.Settings["log_file"] = &grpc_generated_v2.LogSettingsRequest_SettingValue{
		ParameterChoice: &grpc_generated_v2.LogSettingsRequest_SettingValue_StringParam{
			StringParam: request.LogFile,
		},
	}
	logSettingsRequest.Settings["log_info"] = &grpc_generated_v2.LogSettingsRequest_SettingValue{
		ParameterChoice: &grpc_generated_v2.LogSettingsRequest_SettingValue_BoolParam{
			BoolParam: request.LogInfo,
		},
	}
	logSettingsRequest.Settings["log_warning"] = &grpc_generated_v2.LogSettingsRequest_SettingValue{
		ParameterChoice: &grpc_generated_v2.LogSettingsRequest_SettingValue_BoolParam{
			BoolParam: request.LogWarning,
		},
	}
	logSettingsRequest.Settings["log_error"] = &grpc_generated_v2.LogSettingsRequest_SettingValue{
		ParameterChoice: &grpc_generated_v2.LogSettingsRequest_SettingValue_BoolParam{
			BoolParam: request.LogError,
		},
	}
	logSettingsRequest.Settings["log_verbose_level"] = &grpc_generated_v2.LogSettingsRequest_SettingValue{
		ParameterChoice: &grpc_generated_v2.LogSettingsRequest_SettingValue_Uint32Param{
			Uint32Param: uint32(request.LogVerboseLevel),
		},
	}
	logSettingsRequest.Settings["log_format"] = &grpc_generated_v2.LogSettingsRequest_SettingValue{
		ParameterChoice: &grpc_generated_v2.LogSettingsRequest_SettingValue_StringParam{
			StringParam: request.LogFormat,
		},
	}

	_, err := c.client.LogSettings(ctx, logSettingsRequest)
	if err != nil {
		return fmt.Errorf("failed to update log settings: %w", err)
	}

	if c.verbose {
		c.logger.Println("updated log settings")
	}

	return nil
}

func (c *client) GetLogSettings(ctx context.Context, options *options.Options) (*models.LogSettingsResponse, error) {
	resp, err := c.client.LogSettings(ctx, &grpc_generated_v2.LogSettingsRequest{})
	if err != nil {
		return nil, fmt.Errorf("failed to get log settings: %w", err)
	}

	logSettings := &models.LogSettingsResponse{}

	for key, settingValue := range resp.Settings {
		switch v := settingValue.ParameterChoice.(type) {
		case *grpc_generated_v2.LogSettingsResponse_SettingValue_StringParam:
			switch key {
			case "log_file":
				logSettings.LogFile = v.StringParam
			case "log_format":
				logSettings.LogFormat = v.StringParam
			}
		case *grpc_generated_v2.LogSettingsResponse_SettingValue_BoolParam:
			switch key {
			case "log_info":
				logSettings.LogInfo = v.BoolParam
			case "log_warning":
				logSettings.LogWarning = v.BoolParam
			case "log_error":
				logSettings.LogError = v.BoolParam
			}
		case *grpc_generated_v2.LogSettingsResponse_SettingValue_Uint32Param:
			if key == "log_verbose_level" {
				logSettings.LogVerboseLevel = int(v.Uint32Param)
			}
		}
	}

	if c.verbose {
		c.logger.Println(logSettings)
	}

	return logSettings, nil
}

func (c *client) GetSystemSharedMemoryStatus(ctx context.Context, name string, options *options.Options) ([]models.SystemSharedMemoryStatusResponse, error) {
	req := &grpc_generated_v2.SystemSharedMemoryStatusRequest{
		Name: name,
	}

	resp, err := c.client.SystemSharedMemoryStatus(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get system shared memory status for region '%s': %w", name, err)
	}

	var status []models.SystemSharedMemoryStatusResponse
	for _, stat := range resp.Regions {
		status = append(status, models.SystemSharedMemoryStatusResponse{
			Name: stat.Name,
		})
	}

	if c.verbose {
		c.logger.Println(status)
	}

	return status, nil
}

func (c *client) RegisterSystemSharedMemory(ctx context.Context, name, key string, byteSize, offset int, options *options.Options) error {
	request := &grpc_generated_v2.SystemSharedMemoryRegisterRequest{
		Name:     name,
		Key:      key,
		ByteSize: uint64(byteSize),
		Offset:   uint64(offset),
	}

	_, err := c.client.SystemSharedMemoryRegister(ctx, request)
	if err != nil {
		return fmt.Errorf("failed to register system shared memory with name '%s': %w", name, err)
	}

	if c.verbose {
		c.logger.Printf("registered system shared memory with name '%s'\n", name)
	}

	return nil
}

func (c *client) UnregisterSystemSharedMemory(ctx context.Context, name string, options *options.Options) error {
	_, err := c.client.SystemSharedMemoryUnregister(ctx, &grpc_generated_v2.SystemSharedMemoryUnregisterRequest{
		Name: name,
	})
	if err != nil {
		return err
	}

	if c.verbose {
		c.logger.Printf("unregistered system shared memory with name '%s'\n", name)
	}

	return nil
}

func (c *client) GetCUDASharedMemoryStatus(ctx context.Context, name string, options *options.Options) ([]models.CUDASharedMemoryStatusResponse, error) {
	req := &grpc_generated_v2.CudaSharedMemoryStatusRequest{
		Name: name,
	}

	resp, err := c.client.CudaSharedMemoryStatus(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get CUDA shared memory status for name '%s': %w", name, err)
	}

	var status []models.CUDASharedMemoryStatusResponse
	for regionName, regionStatus := range resp.Regions {
		status = append(status, models.CUDASharedMemoryStatusResponse{
			Name:     regionName,
			ByteSize: regionStatus.ByteSize,
			DeviceId: regionStatus.DeviceId,
		})
	}

	if c.verbose {
		c.logger.Println(status)
	}

	return status, nil
}

func (c *client) RegisterCUDASharedMemory(ctx context.Context, name string, rawHandle []byte, deviceID, byteSize int, options *options.Options) error {
	request := &grpc_generated_v2.CudaSharedMemoryRegisterRequest{
		Name:      name,
		RawHandle: rawHandle,
		DeviceId:  int64(deviceID),
		ByteSize:  uint64(byteSize),
	}

	_, err := c.client.CudaSharedMemoryRegister(ctx, request)
	if err != nil {
		return err
	}

	if c.verbose {
		c.logger.Printf("registered CUDA shared memory with name '%s'\n", name)
	}

	return nil
}

func (c *client) UnregisterCUDASharedMemory(ctx context.Context, name string, options *options.Options) error {
	_, err := c.client.CudaSharedMemoryUnregister(ctx, &grpc_generated_v2.CudaSharedMemoryUnregisterRequest{
		Name: name,
	})
	if err != nil {
		return err
	}

	if c.verbose {
		c.logger.Printf("unregistered CUDA shared memory with name '%s'\n", name)
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
	requestWrapper := NewRequestWrapper(modelName, modelVersion, inputs, outputs, options)

	request, err := requestWrapper.PrepareRequest()
	if err != nil {
		return nil, err
	}

	// Make the gRPC call
	resp, err := c.client.ModelInfer(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("failed to perform inference: %w", err)
	}

	// Map the response to the InferResult model
	responseWrapper := NewResponseWrapper(resp)
	return NewInferResult(responseWrapper, c.verbose)
}
