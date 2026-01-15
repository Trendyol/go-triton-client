package base

import (
	"context"

	"github.com/Trendyol/go-triton-client/models"
	"github.com/Trendyol/go-triton-client/options"
)

type Client interface {
	// IsServerLive checks if the server is live.
	IsServerLive(ctx context.Context, options *options.Options) (bool, error)
	// IsServerReady checks if the server is ready.
	IsServerReady(ctx context.Context, options *options.Options) (bool, error)
	// IsModelReady checks if the specified model is ready.
	IsModelReady(ctx context.Context, modelName string, modelVersion string, options *options.Options) (bool, error)
	// GetServerMetadata retrieves server metadata.
	GetServerMetadata(ctx context.Context, options *options.Options) (*models.ServerMetadataResponse, error)
	// GetModelMetadata retrieves metadata for a specific model.
	GetModelMetadata(ctx context.Context, modelName string, modelVersion string, options *options.Options) (*models.ModelMetadataResponse, error)
	// GetModelConfig retrieves the configuration for a specific model.
	GetModelConfig(ctx context.Context, modelName string, modelVersion string, options *options.Options) (*models.ModelConfigResponse, error)
	// GetModelRepositoryIndex retrieves the index of the model repository.
	GetModelRepositoryIndex(ctx context.Context, options *options.Options) ([]models.ModelRepositoryIndexResponse, error)
	// LoadModel loads a model into the server.
	LoadModel(ctx context.Context, modelName string, config string, files map[string][]byte, options *options.Options) error
	// UnloadModel unloads a model from the server.
	UnloadModel(ctx context.Context, modelName string, unloadDependents bool, options *options.Options) error
	// GetInferenceStatistics retrieves inference statistics for a model.
	GetInferenceStatistics(ctx context.Context, modelName string, modelVersion string, options *options.Options) (*models.InferenceStatisticsResponse, error)
	// GetTraceSettings retrieves trace settings for a model or the server.
	GetTraceSettings(ctx context.Context, modelName string, options *options.Options) (*models.TraceSettingsResponse, error)
	// UpdateLogSettings updates the log settings of the server.
	UpdateLogSettings(ctx context.Context, request models.LogSettingsRequest, options *options.Options) error
	// GetLogSettings retrieves the log settings of the server.
	GetLogSettings(ctx context.Context, options *options.Options) (*models.LogSettingsResponse, error)
	// GetSystemSharedMemoryStatus retrieves the status of the system shared memory.
	GetSystemSharedMemoryStatus(ctx context.Context, regionName string, options *options.Options) ([]models.SystemSharedMemoryStatusResponse, error)
	// RegisterSystemSharedMemory registers a region of system shared memory.
	RegisterSystemSharedMemory(ctx context.Context, name string, key string, byteSize int, offset int, options *options.Options) error
	// UnregisterSystemSharedMemory unregisters a region of system shared memory.
	UnregisterSystemSharedMemory(ctx context.Context, name string, options *options.Options) error
	// GetCUDASharedMemoryStatus retrieves the status of the CUDA shared memory.
	GetCUDASharedMemoryStatus(ctx context.Context, regionName string, options *options.Options) ([]models.CUDASharedMemoryStatusResponse, error)
	// RegisterCUDASharedMemory registers a region of CUDA shared memory.
	RegisterCUDASharedMemory(ctx context.Context, name string, rawHandle []byte, deviceID int, byteSize int, options *options.Options) error
	// UnregisterCUDASharedMemory unregisters a region of CUDA shared memory.
	UnregisterCUDASharedMemory(ctx context.Context, name string, options *options.Options) error
	// Infer sends an inference request to the server.
	Infer(
		ctx context.Context,
		modelName string,
		modelVersion string,
		inputs []InferInput,
		outputs []InferOutput,
		options *options.InferOptions,
	) (InferResult, error)

	Close() error
}
