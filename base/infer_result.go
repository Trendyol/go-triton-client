package base

import (
	"fmt"
)

// InferResult defines methods for retrieving various types of output data from an inference result.
type InferResult interface {
	// GetOutput returns the output tensor corresponding to the specified name.
	GetOutput(name string) (InferOutput, error)
	// GetShape returns the shape of the output tensor identified by name.
	GetShape(name string) ([]int64, error)
	// AsInt8Slice returns the output data as a slice of int8 values.
	AsInt8Slice(name string) ([]int8, error)
	// AsInt16Slice returns the output data as a slice of int16 values.
	AsInt16Slice(name string) ([]int16, error)
	// AsInt32Slice returns the output data as a slice of int32 values.
	AsInt32Slice(name string) ([]int32, error)
	// AsInt64Slice returns the output data as a slice of int64 values.
	AsInt64Slice(name string) ([]int64, error)
	// AsUint8Slice returns the output data as a slice of uint8 values.
	AsUint8Slice(name string) ([]uint8, error)
	// AsUint16Slice returns the output data as a slice of uint16 values.
	AsUint16Slice(name string) ([]uint16, error)
	// AsUint32Slice returns the output data as a slice of uint32 values.
	AsUint32Slice(name string) ([]uint32, error)
	// AsUint64Slice returns the output data as a slice of uint64 values.
	AsUint64Slice(name string) ([]uint64, error)
	// AsFloat16Slice returns the output data as a slice of float16 values, converted to float64.
	AsFloat16Slice(name string) ([]float64, error)
	// AsFloat32Slice returns the output data as a slice of float32 values.
	AsFloat32Slice(name string) ([]float32, error)
	// AsFloat64Slice returns the output data as a slice of float64 values.
	AsFloat64Slice(name string) ([]float64, error)
	// AsBoolSlice returns the output data as a slice of boolean values.
	AsBoolSlice(name string) ([]bool, error)
	// AsByteSlice returns the output data as a slice of strings.
	AsByteSlice(name string) ([]string, error)
}

// BaseInferResult provides common fields and methods for InferResult implementations.
type BaseInferResult struct {
	OutputsResponse       InferOutputs
	OutputNameToBufferMap map[string]int
	Buffer                []byte
}

func (r *BaseInferResult) GetOutput(name string) (InferOutput, error) {
	if len(r.OutputsResponse.Outputs) == 0 {
		return nil, fmt.Errorf("no outputs found in result")
	}
	for _, output := range r.OutputsResponse.Outputs {
		if output.GetName() == name {
			return output, nil
		}
	}
	return nil, fmt.Errorf("output %s not found", name)
}

func (r *BaseInferResult) GetShape(name string) ([]int64, error) {
	output, err := r.GetOutput(name)
	if err != nil {
		return nil, err
	}
	return output.GetShape(), nil
}
