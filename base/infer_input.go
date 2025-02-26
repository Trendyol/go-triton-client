package base

import (
	"fmt"
	"github.com/Trendyol/go-triton-client/converter"
)

// InferInput is an interface that defines methods for model inference inputs.
type InferInput interface {
	// GetName returns the name of the input tensor.
	GetName() string
	// GetShape returns the shape of the input tensor.
	GetShape() []int64
	// GetDatatype returns the data type of the input tensor.
	GetDatatype() string
	// GetParameters returns the parameters associated with the input tensor.
	GetParameters() map[string]any
	// GetData returns the data of the input tensor.
	GetData() []any
	// GetRawData returns the raw binary data of the input tensor.
	GetRawData() []byte
	// GetTensor constructs the tensor representation suitable for the HTTP/gRPC inference request.
	GetTensor() any
	GetBinaryData() []byte
	// SetDatatype sets the datatype of the input tensor.
	SetDatatype(datatype string)
	// SetShape sets the shape of the input tensor.
	SetShape(shape []int64)
	// SetData sets the data for the input tensor.
	// If binaryData is true, it serializes the inputTensor and stores it as RawData.
	// If binaryData is false, it flattens the inputTensor and stores it as Data.
	SetData(inputTensor any, binaryData bool) error
}

// BaseInferInput is a base struct that implements common functionality for InferInput.
type BaseInferInput struct {
	Name       string
	Shape      []int64
	Datatype   string
	Parameters map[string]any
	Data       []any
	RawData    []byte
}

func (input *BaseInferInput) GetName() string {
	return input.Name
}

func (input *BaseInferInput) GetShape() []int64 {
	return input.Shape
}

func (input *BaseInferInput) GetDatatype() string {
	return input.Datatype
}

func (input *BaseInferInput) GetParameters() map[string]any {
	return input.Parameters
}

func (input *BaseInferInput) GetData() []any {
	return input.Data
}

func (input *BaseInferInput) GetRawData() []byte {
	return input.RawData
}

func (input *BaseInferInput) SetDatatype(datatype string) {
	input.Datatype = datatype
}

func (input *BaseInferInput) SetShape(shape []int64) {
	input.Shape = shape
}

func (input *BaseInferInput) SetData(inputTensor any, binaryData bool) error {
	// Validate the input tensor type matches the expected datatype
	if input.Datatype != GetDatatype(inputTensor) && input.Datatype != "FP16" {
		return fmt.Errorf("got unexpected datatype %T from input tensor, expected %s", inputTensor, input.Datatype)
	}

	if !binaryData {
		// For non-binary data, remove binary data size parameter and set Data
		delete(input.Parameters, "binary_data_size")
		input.RawData = nil
		input.Data = converter.FlattenData(inputTensor)
	} else {
		// For binary data, serialize the tensor and set RawData
		input.Data = nil
		rawData, err := converter.SerializeTensor(inputTensor)
		if err != nil {
			return err
		}
		input.RawData = rawData
		input.Parameters["binary_data_size"] = len(rawData)
	}

	return nil
}

// GetDatatype determines the data type of the input tensor based on its Go type.
func GetDatatype(inputTensor any) string {
	switch inputTensor.(type) {
	case []int8:
		return "INT8"
	case []int16:
		return "INT16"
	case []int32:
		return "INT32"
	case []int64:
		return "INT64"
	case []uint16:
		return "UINT16"
	case []uint32:
		return "UINT32"
	case []uint64:
		return "UINT64"
	case []float32:
		return "FP32"
	case []float64:
		return "FP64"
	case []byte:
		return "BYTES"
	case []bool:
		return "BOOL"
	case []string:
		return "BYTES"
	default:
		return "UNKNOWN"
	}
}
