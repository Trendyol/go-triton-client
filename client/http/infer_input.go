package http

import (
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/converter"
)

// InferInput is the HTTP implementation of the base.InferInput interface.
type InferInput struct {
	*base.BaseInferInput
}

// NewInferInput creates a new HTTP InferInput instance with the given parameters.
func NewInferInput(name string, datatype string, shape []int64, parameters map[string]interface{}) base.InferInput {
	if parameters == nil {
		parameters = make(map[string]interface{})
	}
	return &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:          name,
			Shape:         shape,
			Datatype:      datatype,
			Parameters:    parameters,
			DataConverter: converter.NewDataConverter(),
		},
	}
}

// GetTensor constructs the tensor representation suitable for the HTTP inference request.
func (input *InferInput) GetTensor() any {
	tensor := map[string]interface{}{
		"name":     input.Name,
		"shape":    input.Shape,
		"datatype": input.Datatype,
	}

	if len(input.Parameters) > 0 {
		tensor["parameters"] = input.Parameters
	}

	if input.Parameters["shared_memory_region"] == nil && (input.RawData == nil || len(input.RawData) == 0) {
		if input.Data != nil {
			tensor["data"] = input.Data
		}
	}

	return tensor
}

// GetBinaryData returns the raw binary data of the input tensor.
func (input *InferInput) GetBinaryData() []byte {
	return input.RawData
}