package grpc

import (
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/Trendyol/go-triton-client/options"
)

// RequestWrapper is a struct that encapsulates all necessary data to create and manage a gRPC request for a model inference.
// It includes model details, inputs, outputs, request-specific configurations, and parameters.
type RequestWrapper struct {
	ModelName    string
	ModelVersion string
	Inputs       []base.InferInput
	Outputs      []base.InferOutput
	Options      *options.InferOptions
}

// NewRequestWrapper initializes and returns a new RequestWrapper instance
// with the provided model details, inputs, outputs, and other configurations.
func NewRequestWrapper(
	modelName, modelVersion string,
	inputs []base.InferInput,
	outputs []base.InferOutput,
	opts *options.InferOptions,
) *RequestWrapper {
	if opts == nil {
		opts = &options.InferOptions{
			Headers:                      nil,
			QueryParams:                  nil,
			RequestID:                    nil,
			SequenceID:                   nil,
			SequenceStart:                nil,
			SequenceEnd:                  nil,
			Priority:                     nil,
			Timeout:                      nil,
			RequestCompressionAlgorithm:  nil,
			ResponseCompressionAlgorithm: nil,
			Parameters:                   nil,
		}
	}
	return &RequestWrapper{
		ModelName:    modelName,
		ModelVersion: modelVersion,
		Inputs:       inputs,
		Outputs:      outputs,
		Options:      opts,
	}
}

// PrepareRequest prepares the gRPC request for model inference.
// It constructs the inference request payload and sets appropriate parameters.
func (w *RequestWrapper) PrepareRequest() (*grpc_generated_v2.ModelInferRequest, error) {
	request, err := w.getInferenceRequest()
	if err != nil {
		return nil, err
	}

	return request, nil
}

// getInferenceRequest constructs the inference request payload,
// including inputs, outputs, and parameters.
func (w *RequestWrapper) getInferenceRequest() (*grpc_generated_v2.ModelInferRequest, error) {
	request := &grpc_generated_v2.ModelInferRequest{
		ModelName:    w.ModelName,
		ModelVersion: w.ModelVersion,
		Parameters:   make(map[string]*grpc_generated_v2.InferParameter),
	}

	if w.Options.RequestID != nil && *w.Options.RequestID != "" {
		request.Id = *w.Options.RequestID
	}

	w.addSequenceParameters(request.Parameters)
	w.addPriorityAndTimeout(request.Parameters)

	inputTensors, rawInputContents := w.convertInputsToTensors()
	request.Inputs = inputTensors
	request.RawInputContents = rawInputContents

	if len(w.Outputs) > 0 {
		outputTensors := w.convertOutputsToTensors()
		request.Outputs = outputTensors
	} else {
		request.Parameters["binary_data_output"] = &grpc_generated_v2.InferParameter{
			ParameterChoice: &grpc_generated_v2.InferParameter_BoolParam{
				BoolParam: true,
			},
		}
	}

	if err := w.addCustomParameters(request.Parameters); err != nil {
		return nil, err
	}

	return request, nil
}

// addSequenceParameters adds sequence-related parameters to the inference request if sequence details (ID, start, end) are provided.
func (w *RequestWrapper) addSequenceParameters(parameters map[string]*grpc_generated_v2.InferParameter) {
	if w.Options.SequenceID != nil {
		parameters["sequence_id"] = &grpc_generated_v2.InferParameter{
			ParameterChoice: &grpc_generated_v2.InferParameter_Int64Param{
				Int64Param: int64(*w.Options.SequenceID),
			},
		}
		if w.Options.SequenceStart != nil {
			parameters["sequence_start"] = &grpc_generated_v2.InferParameter{
				ParameterChoice: &grpc_generated_v2.InferParameter_BoolParam{
					BoolParam: *w.Options.SequenceStart,
				},
			}
		}
		if w.Options.SequenceEnd != nil {
			parameters["sequence_end"] = &grpc_generated_v2.InferParameter{
				ParameterChoice: &grpc_generated_v2.InferParameter_BoolParam{
					BoolParam: *w.Options.SequenceEnd,
				},
			}
		}
	}
}

// addPriorityAndTimeout adds priority and timeout parameters to the inference request if specified.
func (w *RequestWrapper) addPriorityAndTimeout(parameters map[string]*grpc_generated_v2.InferParameter) {
	if w.Options.Priority != nil {
		parameters["priority"] = &grpc_generated_v2.InferParameter{
			ParameterChoice: &grpc_generated_v2.InferParameter_Uint64Param{
				Uint64Param: uint64(*w.Options.Priority),
			},
		}
	}
	if w.Options.Timeout != nil {
		parameters["timeout"] = &grpc_generated_v2.InferParameter{
			ParameterChoice: &grpc_generated_v2.InferParameter_Int64Param{
				Int64Param: int64(*w.Options.Timeout),
			},
		}
	}
}

// addCustomParameters adds any custom parameters to the inference request,
// ensuring that no reserved parameters are overwritten, and returns an error if any reserved parameters are used.
func (w *RequestWrapper) addCustomParameters(parameters map[string]*grpc_generated_v2.InferParameter) error {
	for key, value := range w.Options.Parameters {
		switch key {
		case "sequence_id", "sequence_start", "sequence_end", "priority", "binary_data_output":
			return fmt.Errorf("parameter %q is a reserved parameter and cannot be specified", key)
		default:
			param := &grpc_generated_v2.InferParameter{}
			switch v := value.(type) {
			case string:
				param.ParameterChoice = &grpc_generated_v2.InferParameter_StringParam{
					StringParam: v,
				}
			case bool:
				param.ParameterChoice = &grpc_generated_v2.InferParameter_BoolParam{
					BoolParam: v,
				}
			case int:
				param.ParameterChoice = &grpc_generated_v2.InferParameter_Int64Param{
					Int64Param: int64(v),
				}
			case int64:
				param.ParameterChoice = &grpc_generated_v2.InferParameter_Int64Param{
					Int64Param: v,
				}
			case uint64:
				param.ParameterChoice = &grpc_generated_v2.InferParameter_Uint64Param{
					Uint64Param: v,
				}
			case float64:
				param.ParameterChoice = &grpc_generated_v2.InferParameter_DoubleParam{
					DoubleParam: v,
				}
			default:
				return fmt.Errorf("unsupported parameter type for key %q: %T", key, v)
			}
			parameters[key] = param
		}
	}
	return nil
}

// convertInputsToTensors converts the input base.InferInput instances to a format suitable for the inference request payload.
func (w *RequestWrapper) convertInputsToTensors() ([]*grpc_generated_v2.ModelInferRequest_InferInputTensor, [][]byte) {
	inputTensors := make([]*grpc_generated_v2.ModelInferRequest_InferInputTensor, len(w.Inputs))
	var rawInputContents [][]byte
	for i, input := range w.Inputs {
		inputTensors[i] = input.GetTensor().(*grpc_generated_v2.ModelInferRequest_InferInputTensor)
		if rawData := input.GetBinaryData(); rawData != nil {
			rawInputContents = make([][]byte, 0, len(rawData))
			rawInputContents = append(rawInputContents, rawData)
		}
	}
	return inputTensors, rawInputContents
}

// convertOutputsToTensors converts the output base.InferOutput instances to a format suitable for the inference request payload.
func (w *RequestWrapper) convertOutputsToTensors() []*grpc_generated_v2.ModelInferRequest_InferRequestedOutputTensor {
	outputTensors := make([]*grpc_generated_v2.ModelInferRequest_InferRequestedOutputTensor, len(w.Outputs))
	for i, output := range w.Outputs {
		outputTensors[i] = output.GetTensor().(*grpc_generated_v2.ModelInferRequest_InferRequestedOutputTensor)
	}
	return outputTensors
}
