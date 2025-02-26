package base

import (
	"encoding/json"
	"errors"
	"github.com/Trendyol/go-triton-client/converter"
)

// InferOutputs holds model name, version, and output details.
type InferOutputs struct {
	ModelName    string             `json:"model_name"`
	ModelVersion string             `json:"model_version"`
	Outputs      []*BaseInferOutput `json:"outputs"`
}

// InferOutput interface defines methods for output data.
type InferOutput interface {
	// GetName returns the output name.
	GetName() string
	// GetShape returns the output shape.
	GetShape() []int64
	// GetDatatype returns the datatype of the output.
	GetDatatype() string
	// GetParameters returns the parameters of the output.
	GetParameters() map[string]any
	// GetData returns the data of the output.
	GetData() []any
	GetTensor() any
}

// BaseInferOutput represents basic output properties.
type BaseInferOutput struct {
	Name       string
	Shape      []int64
	Datatype   string
	Parameters map[string]any
	Data       []any
}

// UnmarshalJSON customizes the JSON unmarshaling for InferOutput, handling data conversion
// for different datatypes such as FP32, INT32, BOOL, etc.
func (output *BaseInferOutput) UnmarshalJSON(data []byte) error {
	var tempStruct struct {
		Name           string         `json:"name"`
		Shape          []int64        `json:"shape"`
		Datatype       string         `json:"datatype"`
		Parameters     map[string]any `json:"parameters"`
		Data           []any          `json:"data"`
		Classification int            `json:"classification"`
	}
	if err := json.Unmarshal(data, &tempStruct); err != nil {
		return err
	}
	output.Name = tempStruct.Name
	output.Shape = tempStruct.Shape
	output.Datatype = tempStruct.Datatype
	output.Parameters = tempStruct.Parameters
	switch tempStruct.Datatype {
	case "FP32":
		output.Data = converter.ConvertInterfaceSliceToFloat32SliceAsInterface(tempStruct.Data)
	case "FP64":
		output.Data = converter.ConvertInterfaceSliceToFloat64SliceAsInterface(tempStruct.Data)
	case "INT32":
		output.Data = converter.ConvertInterfaceSliceToInt32SliceAsInterface(tempStruct.Data)
	case "INT64":
		output.Data = converter.ConvertInterfaceSliceToInt64SliceAsInterface(tempStruct.Data)
	case "UINT32":
		output.Data = converter.ConvertInterfaceSliceToUint32SliceAsInterface(tempStruct.Data)
	case "UINT64":
		output.Data = converter.ConvertInterfaceSliceToUint64SliceAsInterface(tempStruct.Data)
	case "BOOL":
		output.Data = converter.ConvertInterfaceSliceToBoolSliceAsInterface(tempStruct.Data)
	case "BYTES":
		convertedData, err := converter.ConvertInterfaceSliceToBytesSliceAsInterface(tempStruct.Data)
		if err != nil {
			return err
		}
		output.Data = convertedData
	default:
		output.Data = tempStruct.Data
	}
	return nil
}

func (output *BaseInferOutput) GetName() string {
	return output.Name
}

func (output *BaseInferOutput) GetShape() []int64 {
	return output.Shape
}

func (output *BaseInferOutput) GetDatatype() string {
	return output.Datatype
}

func (output *BaseInferOutput) GetParameters() map[string]any {
	return output.Parameters
}

func (output *BaseInferOutput) GetData() []any {
	return output.Data
}

func (output *BaseInferOutput) GetTensor() any {
	return errors.New("do not use base GetTensor function")
}
