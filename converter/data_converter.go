package converter

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"unsafe"

	"github.com/x448/float16"
)

// SerializeTensor converts a supported tensor into a byte slice.
func SerializeTensor(inputTensor any) ([]byte, error) {
	var buffer bytes.Buffer
	if err := serializeTensorToWriter(inputTensor, &buffer); err != nil {
		return nil, err
	}
	return buffer.Bytes(), nil
}

// serializeTensorToWriter writes the tensor data into the provided io.Writer.
func serializeTensorToWriter(inputTensor any, w io.Writer) error {
	switch tensor := inputTensor.(type) {
	// Grouping cases for fixed-size numeric types allows for a single, efficient write.
	// binary.Write is optimized to handle slices of these types directly.
	case []int32, []int64, []uint16, []uint32, []uint64, []float32, []float64:
		return binary.Write(w, binary.LittleEndian, tensor)

	// The native `int` type can be 32 or 64 bits depending on the architecture.
	// To ensure consistent serialization, we convert it to a fixed-size type like int64.
	case []int:
		// This still requires a loop, but it's necessary for portability.
		int64Slice := make([]int64, len(tensor))
		for i, v := range tensor {
			int64Slice[i] = int64(v)
		}
		return binary.Write(w, binary.LittleEndian, int64Slice)

	// For bools, we convert to a byte slice first and then write the whole slice.
	case []bool:
		byteSlice := make([]byte, len(tensor))
		for i, v := range tensor {
			if v {
				byteSlice[i] = 1
			}
		}
		_, err := w.Write(byteSlice)
		return err

	// []byte is already in a serializable format.
	case []byte:
		_, err := w.Write(tensor)
		return err

	// Strings are variable-length and must be handled individually.
	// Each string is prefixed with its length.
	case []string:
		for _, str := range tensor {
			strBytes := []byte(str)
			// Write length as int32
			if err := binary.Write(w, binary.LittleEndian, int32(len(strBytes))); err != nil {
				return err
			}
			// Write the actual string bytes
			if _, err := w.Write(strBytes); err != nil {
				return err
			}
		}
		return nil

	default:
		return errors.New("unsupported tensor datatype")
	}
}

// --- Flattening ---

// ToAnySlice is a generic helper function that converts a slice of any type
// into a slice of `any` (`[]any`).
func ToAnySlice[T any](slice []T) []any {
	result := make([]any, len(slice))
	for i, v := range slice {
		result[i] = v
	}
	return result
}

// FlattenData converts a typed tensor slice into a generic []any slice.
func FlattenData(inputTensor any) []any {
	switch tensor := inputTensor.(type) {
	case []int:
		return ToAnySlice(tensor)
	case []int32:
		return ToAnySlice(tensor)
	case []int64:
		return ToAnySlice(tensor)
	case []uint16:
		return ToAnySlice(tensor)
	case []uint32:
		return ToAnySlice(tensor)
	case []uint64:
		return ToAnySlice(tensor)
	case []float32:
		return ToAnySlice(tensor)
	case []float64:
		return ToAnySlice(tensor)
	case []byte:
		return ToAnySlice(tensor)
	case []bool:
		return ToAnySlice(tensor)
	case []string:
		return ToAnySlice(tensor)
	default:
		return nil
	}
}

// --- Deserialization ---

// Numeric is a constraint that includes all standard fixed-size numeric types
// that can be handled directly by binary.Read.
type Numeric interface {
	uint8 | int8 | int16 | int32 | int64 | uint16 | uint32 | uint64 | float32 | float64
}

// DeserializeNumericSlice provides a high-performance way to deserialize a byte buffer
// into a slice of a numeric type N. It reads the entire buffer in one call.
func DeserializeNumericSlice[N Numeric](dataBuffer []byte) ([]N, error) {
	var element N
	elementSize := int(unsafe.Sizeof(element))

	if len(dataBuffer) == 0 {
		return []N{}, nil
	}

	if len(dataBuffer)%elementSize != 0 {
		return nil, fmt.Errorf("data buffer length (%d) is not a multiple of element size (%d)", len(dataBuffer), elementSize)
	}

	count := len(dataBuffer) / elementSize

	// Step 1: Perform the zero-copy conversion to the concrete numeric slice []N.
	return unsafe.Slice((*N)(unsafe.Pointer(&dataBuffer[0])), count), nil
}

func DeserializeBoolTensor(dataBuffer []byte) ([]bool, error) {
	if len(dataBuffer) == 0 {
		return []bool{}, nil
	}

	// Direct conversion without function call overhead
	result := make([]bool, len(dataBuffer))
	for i, b := range dataBuffer {
		result[i] = b != 0
	}
	return result, nil
}

func DeserializeFloat16Tensor(dataBuffer []byte) ([]float64, error) {
	if len(dataBuffer) == 0 {
		return []float64{}, nil
	}
	if len(dataBuffer)%2 != 0 {
		return nil, fmt.Errorf("data buffer length (%d) is not a multiple of 2", len(dataBuffer))
	}

	// Bulk convert bytes to []uint16 using unsafe pointer (zero-copy)
	uint16Count := len(dataBuffer) / 2
	uint16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(&dataBuffer[0])), uint16Count)

	// Convert []uint16 to []float64 in single loop
	result := make([]float64, uint16Count)
	for i, bits := range uint16Slice {
		result[i] = float64(float16.Frombits(bits).Float32())
	}
	return result, nil
}

func DeserializeBF16Tensor(encodedTensor []byte) ([]float32, error) {
	if len(encodedTensor) == 0 {
		return []float32{}, nil
	}
	if len(encodedTensor)%2 != 0 {
		return nil, fmt.Errorf("data buffer length (%d) is not a multiple of 2", len(encodedTensor))
	}

	// Bulk convert bytes to []uint16 using unsafe pointer (zero-copy)
	uint16Count := len(encodedTensor) / 2
	uint16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(&encodedTensor[0])), uint16Count)

	// Convert []uint16 to []float32 in single loop
	result := make([]float32, uint16Count)
	for i, bits := range uint16Slice {
		result[i] = math.Float32frombits(uint32(bits) << 16)
	}
	return result, nil
}

func DeserializeBytesTensor(encodedTensor []byte) ([]string, error) {
	var strs []string
	reader := bytes.NewReader(encodedTensor)
	for reader.Len() > 0 {
		var length int32
		if err := binary.Read(reader, binary.LittleEndian, &length); err != nil {
			return nil, fmt.Errorf("failed to read string length: %w", err)
		}
		if reader.Len() < int(length) {
			return nil, fmt.Errorf("unexpected end of tensor: string length %d exceeds buffer size %d", length, reader.Len())
		}
		strBytes := make([]byte, length)
		if _, err := io.ReadFull(reader, strBytes); err != nil {
			return nil, fmt.Errorf("failed to read string bytes: %w", err)
		}
		strs = append(strs, string(strBytes))
	}
	return strs, nil
}

// --- Interface Slice Conversion ---

// ConvertibleFromFloat64 is a constraint for numeric types that can be converted from a float64.
type ConvertibleFromFloat64 interface {
	int32 | int64 | uint32 | uint64 | float32 | float64
}

// ConvertToNumericSlice is a generic and type-safe function.
// It converts []any to []any where each element is converted to type T.
func ConvertToNumericSlice[T ConvertibleFromFloat64](data []any) ([]any, error) {
	result := make([]T, len(data))
	for i, v := range data {
		switch val := v.(type) {
		case int:
			result[i] = T(val)
		case int32:
			result[i] = T(val)
		case int64:
			result[i] = T(val)
		case uint32:
			result[i] = T(val)
		case uint64:
			result[i] = T(val)
		case float32:
			result[i] = T(val)
		case float64:
			result[i] = T(val)
		default:
			return nil, fmt.Errorf("invalid element at index %d: expected numeric type, got %T", i, v)
		}
	}
	return ToAnySlice[T](result), nil
}

// ConvertToBoolSlice is a specific, type-safe function for booleans.
func ConvertToBoolSlice(data []any) ([]bool, error) {
	result := make([]bool, len(data))
	for i, v := range data {
		val, ok := v.(bool)
		if !ok {
			return nil, fmt.Errorf("invalid element at index %d: expected bool, got %T", i, v)
		}
		result[i] = val
	}
	return result, nil
}

// ConvertToBytesSlice is a specific, type-safe function for bytes from strings.
func ConvertToBytesSlice(data []any) ([][]byte, error) {
	result := make([][]byte, len(data))
	for i, v := range data {
		str, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("invalid element at index %d: expected string, got %T", i, v)
		}
		result[i] = []byte(str)
	}
	return result, nil
}

// Reshape1D converts a flat []T into a 1D slice according to the provided shape.
func Reshape1D[T any](data []T, shape []int64) ([]T, error) {
	if len(shape) != 1 {
		return nil, fmt.Errorf("expected 1D shape, got %d dimensions", len(shape))
	}
	total := int(shape[0])
	if len(data) != total {
		return nil, fmt.Errorf("data length mismatch: expected %d, got %d", total, len(data))
	}
	return data, nil
}

// Reshape2D converts a flat []T into a 2D slice ([][]T) according to the provided shape.
func Reshape2D[T any](data []T, shape []int64) ([][]T, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("expected 2D shape, got %d dimensions", len(shape))
	}
	rows := int(shape[0])
	cols := int(shape[1])
	if len(data) != rows*cols {
		return nil, fmt.Errorf("data length mismatch: expected %d, got %d", rows*cols, len(data))
	}
	res := make([][]T, rows)
	for i := 0; i < rows; i++ {
		res[i] = data[i*cols : (i+1)*cols]
	}
	return res, nil
}

// Reshape3D converts a flat []T into a 3D slice ([][][]T) according to the provided shape.
func Reshape3D[T any](data []T, shape []int64) ([][][]T, error) {
	if len(shape) != 3 {
		return nil, fmt.Errorf("expected 3D shape, got %d dimensions", len(shape))
	}
	d0, d1, d2 := int(shape[0]), int(shape[1]), int(shape[2])
	if len(data) != d0*d1*d2 {
		return nil, fmt.Errorf("data length mismatch: expected %d, got %d", d0*d1*d2, len(data))
	}

	// 1. Allocate the top-level slice (1st allocation).
	res := make([][][]T, d0)

	// 2. Allocate all the middle-level slices in one contiguous block (2nd allocation).
	midSlices := make([][]T, d0*d1)

	// 3. Point the middle slices to the correct locations in the original data.
	// This loop is very fast as slicing doesn't copy the underlying data.
	for i := 0; i < d0*d1; i++ {
		start := i * d2
		end := start + d2
		midSlices[i] = data[start:end]
	}

	// 4. Link the top-level slices to the middle-level slices.
	for i := 0; i < d0; i++ {
		start := i * d1
		end := start + d1
		res[i] = midSlices[start:end]
	}

	return res, nil
}
