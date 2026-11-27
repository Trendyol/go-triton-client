package grpc

import (
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/Trendyol/go-triton-client/converter"
)

// InferResult represents the result of an inference operation using gRPC.
type InferResult struct {
	*base.BaseInferResult
}

// NewInferResult creates a new gRPC InferResult instance.
func NewInferResult(responseWrapper base.ResponseWrapper, verbose bool) (base.InferResult, error) {
	result := base.InferOutputs{}
	var buffer []byte
	outputNameToBufferMap := make(map[string]int, len(result.Outputs))

	response, ok := responseWrapper.GetResponse().(*grpc_generated_v2.ModelInferResponse)
	if !ok {
		return nil, fmt.Errorf("invalid response type")
	}
	result.ModelName = response.ModelName
	result.ModelVersion = response.ModelVersion

	outputs := make([]*base.BaseInferOutput, len(response.Outputs))
	if len(response.Outputs) == 1 && len(response.RawOutputContents) == 1 {
		output := response.Outputs[0]
		buffer = response.RawOutputContents[0]
		outputNameToBufferMap[output.GetName()] = 0
		outputs[0] = &base.BaseInferOutput{
			Name:     output.Name,
			Datatype: output.Datatype,
			Shape:    output.Shape,
		}
	} else {
		totalLen := 0
		for _, b := range response.RawOutputContents {
			totalLen += len(b)
		}
		buffer = make([]byte, 0, totalLen)
		bufferIndex := 0
		for i, output := range response.Outputs {
			dataBuffer := response.RawOutputContents[i]

			modelOutput := &base.BaseInferOutput{
				Name:     output.Name,
				Datatype: output.Datatype,
				Shape:    output.Shape,
			}

			outputNameToBufferMap[output.GetName()] = bufferIndex
			bufferIndex += len(dataBuffer)
			buffer = append(buffer, dataBuffer...)

			outputs[i] = modelOutput
		}
	}

	result.Outputs = outputs

	if verbose {
		fmt.Println("GRPC Response:", response)
		fmt.Println("Buffer:", buffer)
	}

	return &InferResult{
		BaseInferResult: &base.BaseInferResult{
			OutputsResponse:       result,
			OutputNameToBufferMap: outputNameToBufferMap,
			Buffer:                buffer,
		},
	}, nil
}

func (r *InferResult) AsInt8Slice(name string) ([]int8, error) {
	return getAsSlice[int8](name, r, converter.DeserializeNumericSlice[int8])
}

func (r *InferResult) AsInt16Slice(name string) ([]int16, error) {
	return getAsSlice[int16](name, r, converter.DeserializeNumericSlice[int16])
}

func (r *InferResult) AsInt32Slice(name string) ([]int32, error) {
	return getAsSlice[int32](name, r, converter.DeserializeNumericSlice[int32])
}

func (r *InferResult) AsInt64Slice(name string) ([]int64, error) {
	return getAsSlice[int64](name, r, converter.DeserializeNumericSlice[int64])
}

func (r *InferResult) AsUint8Slice(name string) ([]uint8, error) {
	return getAsSlice[uint8](name, r, converter.DeserializeNumericSlice[uint8])
}

func (r *InferResult) AsUint16Slice(name string) ([]uint16, error) {
	return getAsSlice[uint16](name, r, converter.DeserializeNumericSlice[uint16])
}

func (r *InferResult) AsUint32Slice(name string) ([]uint32, error) {
	return getAsSlice[uint32](name, r, converter.DeserializeNumericSlice[uint32])
}

func (r *InferResult) AsUint64Slice(name string) ([]uint64, error) {
	return getAsSlice[uint64](name, r, converter.DeserializeNumericSlice[uint64])
}

func (r *InferResult) AsFloat16Slice(name string) ([]float64, error) {
	return getAsSlice[float64](name, r, converter.DeserializeFloat16Tensor)
}

func (r *InferResult) AsFloat16Slice32(name string) ([]float32, error) {
	return getAsSlice[float32](name, r, converter.DeserializeFloat16Tensor32)
}

func (r *InferResult) AsFloat32Slice(name string) ([]float32, error) {
	return getAsSlice[float32](name, r, converter.DeserializeNumericSlice[float32])
}

func (r *InferResult) AsFloat64Slice(name string) ([]float64, error) {
	return getAsSlice[float64](name, r, converter.DeserializeNumericSlice[float64])
}

func (r *InferResult) AsBoolSlice(name string) ([]bool, error) {
	return getAsSlice[bool](name, r, converter.DeserializeBoolTensor)
}

func (r *InferResult) AsByteSlice(name string) ([]string, error) {
	return getAsSlice[string](name, r, converter.DeserializeBytesTensor)
}

func getAsSlice[T any](name string, inferResult *InferResult, deserializer func(buffer []byte) ([]T, error)) ([]T, error) {
	_, err := inferResult.GetOutput(name)
	if err != nil {
		return nil, err
	}

	startIndex := inferResult.OutputNameToBufferMap[name]
	endIndex := startIndex + len(inferResult.Buffer[startIndex:])
	dataBuffer := inferResult.Buffer[startIndex:endIndex]

	return deserializer(dataBuffer)
}
