package grpc

import (
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"log"
)

// InferInput is the gRPC implementation of the base.InferInput interface.
type InferInput struct {
	*base.BaseInferInput
}

// NewInferInput creates a new gRPC InferInput instance with the given parameters.
func NewInferInput(name string, datatype string, shape []int64, parameters map[string]any) base.InferInput {
	if parameters == nil {
		parameters = make(map[string]any)
	}
	return &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:       name,
			Shape:      shape,
			Datatype:   datatype,
			Parameters: parameters,
		},
	}
}

func (input *InferInput) GetBinaryData() []byte {
	return input.RawData
}

func (input *InferInput) GetTensor() any {
	inputTensor := &grpc_generated_v2.ModelInferRequest_InferInputTensor{
		Name:     input.Name,
		Datatype: input.Datatype,
		Shape:    input.Shape,
	}

	// If raw data is present, return the tensor directly.
	if len(input.RawData) > 0 {
		return inputTensor
	}

	contents := &grpc_generated_v2.InferTensorContents{}
	var err error

	switch input.Datatype {
	case "UINT8":
		var val []uint32
		val, err = processField[uint8, []uint32](input.Data, uInt8SliceToUint32Slice)
		if err == nil {
			contents.UintContents = val
		}
	case "UINT16":
		var val []uint32
		val, err = processField[uint16, []uint32](input.Data, uInt16SliceToUint32Slice)
		if err == nil {
			contents.UintContents = val
		}
	case "UINT32":
		var val []uint32
		val, err = processField[uint32, []uint32](input.Data, identity[[]uint32])
		if err == nil {
			contents.UintContents = val
		}
	case "UINT64":
		var val []uint64
		val, err = processField[uint64, []uint64](input.Data, identity[[]uint64])
		if err == nil {
			contents.Uint64Contents = val
		}
	case "INT8":
		var val []int32
		val, err = processField[int8, []int32](input.Data, int8SliceToInt32Slice)
		if err == nil {
			contents.IntContents = val
		}
	case "INT16":
		var val []int32
		val, err = processField[int16, []int32](input.Data, int16SliceToInt32Slice)
		if err == nil {
			contents.IntContents = val
		}
	case "INT32":
		var val []int32
		val, err = processField[int32, []int32](input.Data, identity[[]int32])
		if err == nil {
			contents.IntContents = val
		}
	case "INT64":
		var val []int64
		val, err = processField[int64, []int64](input.Data, identity[[]int64])
		if err == nil {
			contents.Int64Contents = val
		}
	case "FP32":
		var val []float32
		val, err = processField[float32, []float32](input.Data, identity[[]float32])
		if err == nil {
			contents.Fp32Contents = val
		}
	case "FP64":
		var val []float64
		val, err = processField[float64, []float64](input.Data, identity[[]float64])
		if err == nil {
			contents.Fp64Contents = val
		}
	case "BOOL":
		var val []bool
		val, err = processField[bool, []bool](input.Data, identity[[]bool])
		if err == nil {
			contents.BoolContents = val
		}
	case "BYTES":
		var strSlice []string
		strSlice, err = convertSlice[string](input.Data)
		if err == nil {
			contents.BytesContents = stringsToByteSlices(strSlice)
		}
	default:
		contents = nil
		log.Printf("unsupported datatype: %s", input.Datatype)
	}

	if err != nil {
		contents = nil
		log.Printf("failed to set data: %v", err)
	}
	inputTensor.Contents = contents

	return inputTensor
}

// int8SliceToInt32Slice converts a slice of int8 values to a slice of int32 values.
func int8SliceToInt32Slice(data []int8) []int32 {
	result := make([]int32, len(data))
	for i, v := range data {
		result[i] = int32(v)
	}
	return result
}

// int16SliceToInt32Slice converts a slice of int16 values to a slice of int32 values.
func int16SliceToInt32Slice(data []int16) []int32 {
	result := make([]int32, len(data))
	for i, v := range data {
		result[i] = int32(v)
	}
	return result
}

// uInt8SliceToInt32Slice converts a slice of int8 values to a slice of int32 values.
func uInt8SliceToUint32Slice(data []uint8) []uint32 {
	result := make([]uint32, len(data))
	for i, v := range data {
		result[i] = uint32(v)
	}
	return result
}

// uInt16SliceToInt32Slice converts a slice of int16 values to a slice of int32 values.
func uInt16SliceToUint32Slice(data []uint16) []uint32 {
	result := make([]uint32, len(data))
	for i, v := range data {
		result[i] = uint32(v)
	}
	return result
}

// stringsToByteSlices converts a slice of string values to a slice of []byte values.
func stringsToByteSlices(texts []string) [][]byte {
	result := make([][]byte, len(texts))
	for i, s := range texts {
		result[i] = []byte(s)
	}
	return result
}

// processField converts input.Data (of type []any) into a typed slice using convertSlice,
// then applies the provided conversion function (conv) and returns the result.
func processField[T any, U any](data []any, conv func([]T) U) (U, error) {
	slice, err := convertSlice[T](data)
	if err != nil {
		var zero U
		return zero, err
	}
	return conv(slice), nil
}

// identity is a helper generic function that returns its input unchanged.
func identity[T any](v T) T {
	return v
}

// convertSlice converts a slice of type []any to []T. It returns an error if any element
// in the slice cannot be asserted to type T.
func convertSlice[T any](data []any) ([]T, error) {
	result := make([]T, len(data))
	for i, v := range data {
		converted, ok := v.(T)
		if !ok {
			return nil, fmt.Errorf("cannot convert element at index %d: %T cannot be converted to target type", i, v)
		}
		result[i] = converted
	}
	return result, nil
}
