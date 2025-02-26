package grpc

import (
	"context"
	"errors"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/Trendyol/go-triton-client/mocks"
	"github.com/Trendyol/go-triton-client/models"
	"github.com/Trendyol/go-triton-client/options"
	"github.com/stretchr/testify/assert"
	"go.uber.org/mock/gomock"
	"log"
	"testing"
)

func TestNewClient_Success(t *testing.T) {
	mockCtrl := gomock.NewController(t)
	defer mockCtrl.Finish()

	c, err := NewClient("localhost:50051", true, 30.0, 10.0, false, false, nil, nil)

	assert.NoError(t, err)
	assert.NotNil(t, c)
}

func TestNewClient_Error(t *testing.T) {
	mockCtrl := gomock.NewController(t)
	defer mockCtrl.Finish()

	c, err := NewClient("invalid_url\n", true, 30.0, 10.0, false, false, nil, nil)

	assert.Error(t, err)
	assert.Nil(t, c)
}

func TestIsServerLive(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().ServerLive(gomock.Any(), gomock.Any()).Return(&grpc_generated_v2.ServerLiveResponse{Live: true}, nil)

	c := &client{
		client: mockClient,
	}

	live, err := c.IsServerLive(context.Background(), &options.Options{})
	assert.NoError(t, err)
	assert.True(t, live)
}

func TestIsServerReady(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().ServerReady(gomock.Any(), gomock.Any()).Return(&grpc_generated_v2.ServerReadyResponse{Ready: true}, nil)

	c := &client{
		client: mockClient,
	}

	ready, err := c.IsServerReady(context.Background(), &options.Options{})
	assert.NoError(t, err)
	assert.True(t, ready)
}

func TestIsModelReady(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "test_model"
	modelVersion := "1"

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().ModelReady(gomock.Any(), &grpc_generated_v2.ModelReadyRequest{Name: modelName, Version: modelVersion}).Return(&grpc_generated_v2.ModelReadyResponse{Ready: true}, nil)

	c := &client{
		client: mockClient,
	}

	ready, err := c.IsModelReady(context.Background(), modelName, modelVersion, &options.Options{})
	assert.NoError(t, err)
	assert.True(t, ready)
}

func TestGetServerMetadata(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resp := &grpc_generated_v2.ServerMetadataResponse{
		Name:       "triton",
		Version:    "2.0",
		Extensions: []string{"inference"},
	}

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().ServerMetadata(gomock.Any(), gomock.Any()).Return(resp, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	metadata, err := c.GetServerMetadata(context.Background(), &options.Options{})
	assert.NoError(t, err)
	assert.Equal(t, resp.Name, metadata.Name)
	assert.Equal(t, resp.Version, metadata.Version)
	assert.Equal(t, resp.Extensions, metadata.Extensions)
}

func TestGetModelMetadata(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "test_model"
	modelVersion := "1"

	resp := &grpc_generated_v2.ModelMetadataResponse{
		Name:     modelName,
		Versions: []string{modelVersion},
		Platform: "tensorflow",
		Inputs: []*grpc_generated_v2.ModelMetadataResponse_TensorMetadata{
			{
				Name:     "input0",
				Datatype: "FP32",
				Shape:    []int64{1, 3, 224, 224},
			},
		},
		Outputs: []*grpc_generated_v2.ModelMetadataResponse_TensorMetadata{
			{
				Name:     "output0",
				Datatype: "FP32",
				Shape:    []int64{1, 1000},
			},
		},
	}

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().ModelMetadata(gomock.Any(), gomock.Any()).Return(resp, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	metadata, err := c.GetModelMetadata(context.Background(), modelName, modelVersion, &options.Options{})
	assert.NoError(t, err)
	assert.Equal(t, resp.Name, metadata.Name)
	assert.Equal(t, resp.Platform, metadata.Platform)
}

func TestGetModelConfig(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "test_model"
	modelVersion := "1"

	resp := &grpc_generated_v2.ModelConfigResponse{
		Config: &grpc_generated_v2.ModelConfig{
			Name:     modelName,
			Platform: "tensorflow",
			Backend:  "",
			Runtime:  "",
			VersionPolicy: &grpc_generated_v2.ModelVersionPolicy{
				PolicyChoice: &grpc_generated_v2.ModelVersionPolicy_Latest_{
					Latest: &grpc_generated_v2.ModelVersionPolicy_Latest{
						NumVersions: 1,
					},
				},
			},
			MaxBatchSize: 0,
			Input: []*grpc_generated_v2.ModelInput{
				{
					Name:     "input0",
					DataType: grpc_generated_v2.DataType_TYPE_FP32,
					Dims:     []int64{1, 3, 224, 224},
				},
			},
			Output: []*grpc_generated_v2.ModelOutput{
				{
					Name:     "output0",
					DataType: grpc_generated_v2.DataType_TYPE_FP32,
					Dims:     []int64{1, 1000},
				},
			},
			BatchInput:       nil,
			BatchOutput:      nil,
			Optimization:     nil,
			SchedulingChoice: nil,
			InstanceGroup: []*grpc_generated_v2.ModelInstanceGroup{
				{
					Name:             "",
					Kind:             0,
					Count:            0,
					RateLimiter:      nil,
					Gpus:             []int32{1, 2},
					SecondaryDevices: nil,
					Profile:          nil,
					Passive:          false,
					HostPolicy:       "",
				},
			},
			DefaultModelFilename:   "",
			CcModelFilenames:       nil,
			MetricTags:             nil,
			Parameters:             nil,
			ModelWarmup:            nil,
			ModelOperations:        nil,
			ModelTransactionPolicy: nil,
			ModelRepositoryAgents:  nil,
			ResponseCache:          nil,
		},
	}

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().ModelConfig(gomock.Any(), gomock.Any()).Return(resp, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	config, err := c.GetModelConfig(context.Background(), modelName, modelVersion, &options.Options{})
	assert.NoError(t, err)
	assert.Equal(t, resp.Config.Name, config.Name)
	assert.Equal(t, resp.Config.Platform, config.Platform)
}

func TestGetModelRepositoryIndex(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resp := &grpc_generated_v2.RepositoryIndexResponse{
		Models: []*grpc_generated_v2.RepositoryIndexResponse_ModelIndex{
			{
				Name:    "model1",
				Version: "1",
				State:   "READY",
				Reason:  "",
			},
		},
	}

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().RepositoryIndex(gomock.Any(), gomock.Any()).Return(resp, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	index, err := c.GetModelRepositoryIndex(context.Background(), &options.Options{})
	assert.NoError(t, err)
	assert.Len(t, index, 1)
	assert.Equal(t, resp.Models[0].Name, index[0].Name)
}

func TestLoadModel(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "model1"

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().RepositoryModelLoad(gomock.Any(), gomock.Any()).Return(&grpc_generated_v2.RepositoryModelLoadResponse{}, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	err := c.LoadModel(context.Background(), modelName, "test-config", map[string][]byte{"test-key": []byte("test-data")}, &options.Options{})
	assert.NoError(t, err)
}

func TestUnloadModel(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "model1"

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().RepositoryModelUnload(gomock.Any(), gomock.Any()).Return(&grpc_generated_v2.RepositoryModelUnloadResponse{}, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	err := c.UnloadModel(context.Background(), modelName, true, &options.Options{})
	assert.NoError(t, err)
}

func TestGetInferenceStatistics(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "model1"
	modelVersion := "1"

	resp := &grpc_generated_v2.ModelStatisticsResponse{
		ModelStats: []*grpc_generated_v2.ModelStatistics{
			{
				Name:           modelName,
				Version:        modelVersion,
				LastInference:  5,
				InferenceCount: 100,
				ExecutionCount: 50,
				InferenceStats: &grpc_generated_v2.InferStatistics{
					Success: &grpc_generated_v2.StatisticDuration{
						Count: 100,
					},
				},
			},
		},
	}

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().ModelStatistics(gomock.Any(), gomock.Any()).Return(resp, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	stats, err := c.GetInferenceStatistics(context.Background(), modelName, modelVersion, &options.Options{})
	assert.NoError(t, err)
	assert.Len(t, stats.ModelStats, 1)
	assert.Equal(t, modelName, stats.ModelStats[0].Name)
}

func TestGetTraceSettings(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "model1"

	resp := &grpc_generated_v2.TraceSettingResponse{
		Settings: map[string]*grpc_generated_v2.TraceSettingResponse_SettingValue{
			"trace_level":   {Value: []string{"MIN"}},
			"trace_rate":    {Value: []string{"traceRate"}},
			"trace_count":   {Value: []string{"traceCount"}},
			"log_frequency": {Value: []string{"logFrequency"}},
			"trace_file":    {Value: []string{"traceFile"}},
		},
	}

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().TraceSetting(gomock.Any(), gomock.Any()).Return(resp, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	settings, err := c.GetTraceSettings(context.Background(), modelName, &options.Options{})
	assert.NoError(t, err)
	assert.Equal(t, "MIN", settings.TraceLevel[0])
	assert.Equal(t, "traceRate", settings.TraceRate)
	assert.Equal(t, "traceCount", settings.TraceCount)
	assert.Equal(t, "logFrequency", settings.LogFrequency)
	assert.Equal(t, "traceFile", settings.TraceFile)
}

func TestUpdateLogSettings(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	request := models.LogSettingsRequest{
		LogFile:         "server.log",
		LogInfo:         true,
		LogWarning:      true,
		LogError:        true,
		LogVerboseLevel: 1,
		LogFormat:       "default",
	}

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().LogSettings(gomock.Any(), gomock.Any()).Return(&grpc_generated_v2.LogSettingsResponse{}, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	err := c.UpdateLogSettings(context.Background(), request, &options.Options{})
	assert.NoError(t, err)
}

func TestGetLogSettings(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resp := &grpc_generated_v2.LogSettingsResponse{
		Settings: map[string]*grpc_generated_v2.LogSettingsResponse_SettingValue{
			"log_file":          {ParameterChoice: &grpc_generated_v2.LogSettingsResponse_SettingValue_StringParam{StringParam: "server.log"}},
			"log_format":        {ParameterChoice: &grpc_generated_v2.LogSettingsResponse_SettingValue_StringParam{StringParam: "%s %s"}},
			"log_info":          {ParameterChoice: &grpc_generated_v2.LogSettingsResponse_SettingValue_BoolParam{BoolParam: true}},
			"log_warning":       {ParameterChoice: &grpc_generated_v2.LogSettingsResponse_SettingValue_BoolParam{BoolParam: true}},
			"log_error":         {ParameterChoice: &grpc_generated_v2.LogSettingsResponse_SettingValue_BoolParam{BoolParam: true}},
			"log_verbose_level": {ParameterChoice: &grpc_generated_v2.LogSettingsResponse_SettingValue_Uint32Param{Uint32Param: 1}},
		},
	}

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().LogSettings(gomock.Any(), gomock.Any()).Return(resp, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	settings, err := c.GetLogSettings(context.Background(), &options.Options{})
	assert.NoError(t, err)
	assert.Equal(t, "server.log", settings.LogFile)
	assert.True(t, settings.LogInfo)
}

func TestGetSystemSharedMemoryStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	name := "shared_memory_region"

	resp := &grpc_generated_v2.SystemSharedMemoryStatusResponse{
		Regions: map[string]*grpc_generated_v2.SystemSharedMemoryStatusResponse_RegionStatus{
			name: {Name: name},
		},
	}

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().SystemSharedMemoryStatus(gomock.Any(), gomock.Any()).Return(resp, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	status, err := c.GetSystemSharedMemoryStatus(context.Background(), name, &options.Options{})
	assert.NoError(t, err)
	assert.Len(t, status, 1)
	assert.Equal(t, name, status[0].Name)
}

func TestRegisterSystemSharedMemory(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	name := "shared_memory_region"

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().SystemSharedMemoryRegister(gomock.Any(), gomock.Any()).Return(&grpc_generated_v2.SystemSharedMemoryRegisterResponse{}, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	err := c.RegisterSystemSharedMemory(context.Background(), name, "key", 1024, 0, &options.Options{})
	assert.NoError(t, err)
}

func TestUnregisterSystemSharedMemory(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	name := "shared_memory_region"

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().SystemSharedMemoryUnregister(gomock.Any(), gomock.Any()).Return(&grpc_generated_v2.SystemSharedMemoryUnregisterResponse{}, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	err := c.UnregisterSystemSharedMemory(context.Background(), name, &options.Options{})
	assert.NoError(t, err)
}

func TestGetCUDASharedMemoryStatus(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	name := "cuda_shared_memory_region"

	resp := &grpc_generated_v2.CudaSharedMemoryStatusResponse{
		Regions: map[string]*grpc_generated_v2.CudaSharedMemoryStatusResponse_RegionStatus{
			name: {
				ByteSize: 1024,
				DeviceId: 0,
			},
		},
	}

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().CudaSharedMemoryStatus(gomock.Any(), gomock.Any()).Return(resp, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	status, err := c.GetCUDASharedMemoryStatus(context.Background(), name, &options.Options{})
	assert.NoError(t, err)
	assert.Len(t, status, 1)
	assert.Equal(t, name, status[0].Name)
}

func TestRegisterCUDASharedMemory(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	name := "cuda_shared_memory_region"

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().CudaSharedMemoryRegister(gomock.Any(), gomock.Any()).Return(&grpc_generated_v2.CudaSharedMemoryRegisterResponse{}, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	err := c.RegisterCUDASharedMemory(context.Background(), name, []byte{0x00}, 0, 1024, &options.Options{})
	assert.NoError(t, err)
}

func TestUnregisterCUDASharedMemory(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	name := "cuda_shared_memory_region"

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().CudaSharedMemoryUnregister(gomock.Any(), gomock.Any()).Return(&grpc_generated_v2.CudaSharedMemoryUnregisterResponse{}, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
		logger:  log.Default(),
	}

	err := c.UnregisterCUDASharedMemory(context.Background(), name, &options.Options{})
	assert.NoError(t, err)
}

func TestInfer(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "model1"
	modelVersion := "1"

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().ModelInfer(gomock.Any(), gomock.Any()).Return(&grpc_generated_v2.ModelInferResponse{}, nil)

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	inputs := []base.InferInput{
		NewInferInput("input0", "FP32", []int64{1, 3, 224, 224}, nil),
	}
	outputs := []base.InferOutput{
		NewInferOutput("output0", nil),
	}

	result, err := c.Infer(
		context.Background(),
		modelName,
		modelVersion,
		inputs,
		outputs,
		nil,
	)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestIsServerLive_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().ServerLive(gomock.Any(), gomock.Any()).Return(nil, errors.New("server live error"))

	c := &client{
		client: mockClient,
	}

	live, err := c.IsServerLive(context.Background(), &options.Options{})
	assert.Error(t, err)
	assert.False(t, live)
}

func TestIsServerReady_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().ServerReady(gomock.Any(), gomock.Any()).Return(nil, errors.New("server ready error"))

	c := &client{
		client: mockClient,
	}

	ready, err := c.IsServerReady(context.Background(), &options.Options{})
	assert.Error(t, err)
	assert.False(t, ready)
}

func TestIsModelReady_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "test_model"
	modelVersion := "1"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)

	mockClient.EXPECT().ModelReady(gomock.Any(), &grpc_generated_v2.ModelReadyRequest{
		Name:    modelName,
		Version: modelVersion,
	}).Return(nil, errors.New("model ready error"))

	c := &client{
		client: mockClient,
	}

	ready, err := c.IsModelReady(context.Background(), modelName, modelVersion, &options.Options{})
	assert.Error(t, err)
	assert.False(t, ready)
}

func TestGetServerMetadata_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().ServerMetadata(gomock.Any(), gomock.Any()).Return(nil, errors.New("server metadata error"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	metadata, err := c.GetServerMetadata(context.Background(), &options.Options{})
	assert.Error(t, err)
	assert.Nil(t, metadata)
}

func TestGetModelMetadata_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "test_model"
	modelVersion := "1"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)

	mockClient.EXPECT().ModelMetadata(gomock.Any(), &grpc_generated_v2.ModelMetadataRequest{
		Name:    modelName,
		Version: modelVersion,
	}).Return(nil, errors.New("model metadata error"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	metadata, err := c.GetModelMetadata(context.Background(), modelName, modelVersion, &options.Options{})
	assert.Error(t, err)
	assert.Nil(t, metadata)
}

func TestGetModelConfig_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "test_model"
	modelVersion := "1"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)

	mockClient.EXPECT().ModelConfig(gomock.Any(), &grpc_generated_v2.ModelConfigRequest{
		Name:    modelName,
		Version: modelVersion,
	}).Return(nil, errors.New("model config error"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	config, err := c.GetModelConfig(context.Background(), modelName, modelVersion, &options.Options{})
	assert.Error(t, err)
	assert.Nil(t, config)
}

func TestGetModelRepositoryIndex_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().RepositoryIndex(gomock.Any(), gomock.Any()).Return(nil, errors.New("repository index error"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	index, err := c.GetModelRepositoryIndex(context.Background(), &options.Options{})
	assert.Error(t, err)
	assert.Nil(t, index)
}

func TestLoadModel_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "model1"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)

	mockClient.EXPECT().RepositoryModelLoad(gomock.Any(), gomock.Any()).Return(nil, errors.New("load model error"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	err := c.LoadModel(context.Background(), modelName, "", nil, &options.Options{})
	assert.Error(t, err)
}

func TestUnloadModel_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "model1"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)

	mockClient.EXPECT().RepositoryModelUnload(gomock.Any(), gomock.Any()).Return(nil, errors.New("unload model error"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	err := c.UnloadModel(context.Background(), modelName, true, &options.Options{})
	assert.Error(t, err)
}

func TestGetInferenceStatistics_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "model1"
	modelVersion := "1"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)

	mockClient.EXPECT().ModelStatistics(gomock.Any(), &grpc_generated_v2.ModelStatisticsRequest{
		Name:    modelName,
		Version: modelVersion,
	}).Return(nil, errors.New("inference statistics error"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	stats, err := c.GetInferenceStatistics(context.Background(), modelName, modelVersion, &options.Options{})
	assert.Error(t, err)
	assert.Nil(t, stats)
}

func TestInfer_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "model1"
	modelVersion := "1"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)

	mockClient.EXPECT().ModelInfer(gomock.Any(), gomock.Any()).Return(nil, errors.New("inference error"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	inputs := []base.InferInput{
		NewInferInput("input0", "FP32", []int64{1, 3, 224, 224}, nil),
	}
	outputs := []base.InferOutput{
		NewInferOutput("output0", nil),
	}

	result, err := c.Infer(
		context.Background(),
		modelName,
		modelVersion,
		inputs,
		outputs,
		nil,
	)

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestGetTraceSettings_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	modelName := "test_model"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().TraceSetting(gomock.Any(), gomock.Any()).Return(nil, errors.New("failed to get trace settings"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	traceSettings, err := c.GetTraceSettings(context.Background(), modelName, &options.Options{})
	assert.Error(t, err)
	assert.Nil(t, traceSettings)
}

func TestUpdateLogSettings_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().LogSettings(gomock.Any(), gomock.Any()).Return(nil, errors.New("failed to update log settings"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	logSettingsReq := models.LogSettingsRequest{
		LogFile:         "test.log",
		LogInfo:         true,
		LogWarning:      true,
		LogError:        true,
		LogVerboseLevel: 2,
		LogFormat:       "json",
	}

	err := c.UpdateLogSettings(context.Background(), logSettingsReq, &options.Options{})
	assert.Error(t, err)
}

func TestGetLogSettings_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().LogSettings(gomock.Any(), gomock.Any()).Return(nil, errors.New("failed to get log settings"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	logSettings, err := c.GetLogSettings(context.Background(), &options.Options{})
	assert.Error(t, err)
	assert.Nil(t, logSettings)
}

func TestGetSystemSharedMemoryStatus_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	regionName := "test_region"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().SystemSharedMemoryStatus(gomock.Any(), gomock.Any()).Return(nil, errors.New("failed to get system shared memory status"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	status, err := c.GetSystemSharedMemoryStatus(context.Background(), regionName, &options.Options{})
	assert.Error(t, err)
	assert.Nil(t, status)
}

func TestRegisterSystemSharedMemory_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	regionName := "test_region"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().SystemSharedMemoryRegister(gomock.Any(), gomock.Any()).Return(nil, errors.New("failed to register system shared memory"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	err := c.RegisterSystemSharedMemory(context.Background(), regionName, "key", 1024, 0, &options.Options{})
	assert.Error(t, err)
}

func TestUnregisterSystemSharedMemory_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	regionName := "test_region"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().SystemSharedMemoryUnregister(gomock.Any(), gomock.Any()).Return(nil, errors.New("failed to unregister system shared memory"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	err := c.UnregisterSystemSharedMemory(context.Background(), regionName, &options.Options{})
	assert.Error(t, err)
}

func TestGetCUDASharedMemoryStatus_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	cudaMemoryName := "cuda_region"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().CudaSharedMemoryStatus(gomock.Any(), gomock.Any()).Return(nil, errors.New("failed to get CUDA shared memory status"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	status, err := c.GetCUDASharedMemoryStatus(context.Background(), cudaMemoryName, &options.Options{})
	assert.Error(t, err)
	assert.Nil(t, status)
}

func TestRegisterCUDASharedMemory_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	cudaMemoryName := "cuda_region"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().CudaSharedMemoryRegister(gomock.Any(), gomock.Any()).Return(nil, errors.New("failed to register CUDA shared memory"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	err := c.RegisterCUDASharedMemory(context.Background(), cudaMemoryName, []byte{}, 1, 1024, &options.Options{})
	assert.Error(t, err)
}

func TestUnregisterCUDASharedMemory_NotSuccessResponse(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	cudaMemoryName := "cuda_region"
	mockClient := mocks.NewMockGRPCInferenceServiceClient(ctrl)
	mockClient.EXPECT().CudaSharedMemoryUnregister(gomock.Any(), gomock.Any()).Return(nil, errors.New("failed to unregister CUDA shared memory"))

	c := &client{
		client:  mockClient,
		verbose: true,
	}

	err := c.UnregisterCUDASharedMemory(context.Background(), cudaMemoryName, &options.Options{})
	assert.Error(t, err)
}

func TestInfer_PrepareRequestError(t *testing.T) {
	c := &client{}

	invalidInputs := []base.InferInput{
		NewInferInput("", "FP32", []int64{}, nil),
	}
	outputs := []base.InferOutput{
		NewInferOutput("output0", nil),
	}

	_, err := c.Infer(
		context.Background(),
		"model_name",
		"model_version",
		invalidInputs,
		outputs,
		&options.InferOptions{
			Parameters: map[string]any{"test": complex(1, 2)},
		},
	)

	assert.Error(t, err)
}
