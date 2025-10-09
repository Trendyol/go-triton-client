# Go Triton Client [![Go Report Card](https://goreportcard.com/badge/github.com/Trendyol/go-triton-client)](https://goreportcard.com/report/github.com/Trendyol/go-triton-client) ![Latest Release](https://img.shields.io/github/v/release/Trendyol/go-triton-client) ![Go Reference](https://pkg.go.dev/badge/github.com/Trendyol/go-triton-client.svg) ![License](https://img.shields.io/github/license/Trendyol/go-triton-client)

# Introduction
The Go Triton Client is a robust and versatile Software Development Kit (SDK) designed to facilitate seamless integration with NVIDIA's Triton Inference Server. This SDK provides both gRPC and HTTP clients, enabling Go developers to interact with Triton for deploying and managing machine learning models with ease.

# What is Triton Inference Server?
NVIDIA Triton Inference Server is an open-source inference serving software that simplifies the deployment of AI models at scale. Triton supports models from multiple frameworks (TensorFlow, PyTorch, ONNX Runtime, etc.) and provides features such as dynamic batching, concurrent model execution, and model optimization to maximize the utilization of compute resources.

# Why Use the Go Triton Client?
The Go Triton Client allows Go developers to interact with the Triton Inference Server directly from Go applications. With support for both HTTP and gRPC protocols, it provides a flexible and efficient way to manage models and perform inference requests, leveraging Go's concurrency features and performance benefits.

# Related GitHub Repositories
 - NVIDIA Triton Inference Server: https://github.com/triton-inference-server/server

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Creating a Client](#creating-a-client)
    - [HTTP Client](#http-client)
    - [gRPC Client](#grpc-client)
  - [Server Health Checks](#server-health-checks)
  - [Inference](#inference)
    - [Performing Inference](#performing-inference)
    - [Handling Different Data Types](#handling-different-data-types)
    - [Adding Custom Parameters](#adding-custom-parameters)
  - [Examples](#examples)
  - [End-to-End Example with Triton Inference Server](#end-to-end-example-with-triton-inference-server)
- [Contributing](#contributing)

## Features

- **Dual Protocol Support**: Interact with Triton using both gRPC and HTTP protocols.
- **Comprehensive Model Management**: Load, unload, and query models effortlessly.
- **Inference Requests**: Perform synchronous and asynchronous inferences with customizable parameters.
- **Health Monitoring**: Check server and model readiness and liveness.
- **Logging Control**: Retrieve and update server logging settings.
- **Extensible Design**: Easily extendable to support additional Triton features.

## Prerequisites

- **Go**: Ensure you have Go installed (version 1.16 or higher is recommended). [Download Go](https://golang.org/dl/)
- **Triton Inference Server**: Running instance of Triton Inference Server. [Triton Installation Guide](https://github.com/triton-inference-server/server#installation)

## Installation

You can install the Go Triton Client using `go get`:

```bash
go get github.com/Trendyol/go-triton-client
```

## Usage
### Creating a Client
The SDK provides two types of clients: gRPC and HTTP. Choose the one that best fits your application's communication needs.

#### HTTP Client
```go
import (
    "github.com/Trendyol/go-triton-client/client/http"
    "log"
)

func createHTTPClient() {
    client, err := http.NewClient(
        "triton-server.example.com", // Triton HTTP endpoint (without scheme)
        false,                       // verbose logging
        3000,                      // connection timeout in second
        3000,                      // network timeout in second
        true,                        // use SSL
        true,                        // insecure connection
        nil,                         // custom HTTP client (optional)
        nil,                         // logger (optional)
    )
    if err != nil {
        log.Fatalf("Failed to create HTTP client: %v", err)
    }
    // Use the client...
}
```

#### gRPC Client
```go
import (
    "github.com/Trendyol/go-triton-client/client/grpc"
    "log"
)

func createGRPCClient() {
    client, err := grpc.NewClient(
        "triton-server.example.com:8001", // Triton gRPC endpoint
        false,                            // verbose logging
        3,                           // connection timeout in second
        3,                           // network timeout in second
        true,                             // use SSL
        true,                             // insecure connection
        nil,                              // keep-alive parameters
        nil,                              // logger (optional)
    )
    if err != nil {
        log.Fatalf("Failed to create gRPC client: %v", err)
    }
    // Use the client...
}
```

### Server Health Checks
Before performing any operations, it's good practice to check the server's health.

```go
import (
    "context"
    "fmt"
    "github.com/Trendyol/go-triton-client/client/grpc" // or "github.com/Trendyol/go-triton-client/client/http"
    "log"
    "github.com/Trendyol/go-triton-client/options"
)

func checkServerHealth(tritonClient base.Client) {
    isLive, err := tritonClient.IsServerLive(context.Background(), &options.IsServerLiveOptions{})
    if err != nil {
        log.Fatalf("Error checking server liveness: %v", err)
    }
    fmt.Printf("Server is live: %v\n", isLive)

    isReady, err := tritonClient.IsServerReady(context.Background(), &options.IsServerReadyOptions{})
    if err != nil {
        log.Fatalf("Error checking server readiness: %v", err)
    }
    fmt.Printf("Server is ready: %v\n", isReady)
}
```

### Inference
Performing inference involves preparing input data, specifying desired outputs, and handling the response.

#### HTTP Client Inference Example
```go
package main

import (
  "context"
  "fmt"
  "github.com/Trendyol/go-triton-client/base"
  "github.com/Trendyol/go-triton-client/client/http"
  "github.com/Trendyol/go-triton-client/converter"
  "github.com/Trendyol/go-triton-client/postprocess"
  "log"
)

func main() {
  client, err := http.NewClient(
    "triton-server.example.com",
    false,
    200,
    200,
    true,
    true,
    nil,
    nil,
  )
  if err != nil {
    log.Fatalf("Failed to create HTTP client: %v", err)
  }

  // Perform inference
  performInference(client)
}

func performInference(tritonClient base.Client) {
  inputIds := http.NewInferInput("input_ids", "INT64", []int64{2, 3}, nil)
  err := inputIds.SetData([]int{101, 202536, 102, 101, 202536, 102}, true)
  if err != nil {
    log.Fatal(err)
  }
  tokenTypeIds := http.NewInferInput("token_type_ids", "INT64", []int64{2, 3}, nil)
  err = tokenTypeIds.SetData([]int{0, 0, 0, 0, 0, 0}, true)
  if err != nil {
    log.Fatal(err)
  }
  attentionMask := http.NewInferInput("attention_mask", "INT64", []int64{2, 3}, nil)
  err = attentionMask.SetData([]int{1, 1, 1, 1, 1, 1}, true)
  if err != nil {
    log.Fatal(err)
  }

  outputs := []base.InferOutput{
    http.NewInferOutput("logits", map[string]any{"binary_data": true}),
  }

  response, err := tritonClient.Infer(
    context.Background(),
    "ty_bert",
    "1",
    []base.InferInput{inputIds, tokenTypeIds, attentionMask},
    outputs,
    nil,
  )
  if err != nil {
    log.Fatal(err)
  }

  sliceResp, err := response.AsFloat16Slice("logits")
  if err != nil {
    log.Fatal(err)
  }
  shape, err := response.GetShape("logits")
  if err != nil {
    log.Fatalf("Failed to extract shape from response: %v", err)
  }

  // Reshape the logits
  reshapedEmbeddings, err := converter.Reshape3D[float64](sliceResp, shape)
  if err != nil {
    log.Fatal("Failed to parse inference response")
  }

  postprocessManager := postprocess.NewPostprocessManager()
  meanPooledEmbeddings, err := postprocessManager.MeanPoolingFloat64Slice3D(reshapedEmbeddings, [][]int64{{1, 1, 1}, {1, 1, 1}})
  if err != nil {
    log.Fatal(err)
  }
  fmt.Println(meanPooledEmbeddings)
}
```

#### gRPC Client Inference Example
```go
package main

import (
  "context"
  "fmt"
  "github.com/Trendyol/go-triton-client/base"
  "github.com/Trendyol/go-triton-client/client/grpc"
  "github.com/Trendyol/go-triton-client/converter"
  "github.com/Trendyol/go-triton-client/postprocess"
  "log"
)

func main() {
  client, err := http.NewClient(
    "triton-server.example.com",
    false,
    200,
    200,
    true,
    true,
    nil,
    nil,
  )
  if err != nil {
    log.Fatalf("Failed to create HTTP client: %v", err)
  }

  // Perform inference
  performInference(client)
}

func performInference(tritonClient base.Client) {
  inputIds := http.NewInferInput("input_ids", "INT64", []int64{2, 3}, nil)
  err := inputIds.SetData([]int{101, 202536, 102, 101, 202536, 102}, true)
  if err != nil {
    log.Fatal(err)
  }
  tokenTypeIds := http.NewInferInput("token_type_ids", "INT64", []int64{2, 3}, nil)
  err = tokenTypeIds.SetData([]int{0, 0, 0, 0, 0, 0}, true)
  if err != nil {
    log.Fatal(err)
  }
  attentionMask := http.NewInferInput("attention_mask", "INT64", []int64{2, 3}, nil)
  err = attentionMask.SetData([]int{1, 1, 1, 1, 1, 1}, true)
  if err != nil {
    log.Fatal(err)
  }

  outputs := []base.InferOutput{
    http.NewInferOutput("logits", map[string]any{"binary_data": true}),
  }

  response, err := tritonClient.Infer(
    context.Background(),
    "ty_bert",
    "1",
    []base.InferInput{inputIds, tokenTypeIds, attentionMask},
    outputs,
    nil,
  )
  if err != nil {
    log.Fatal(err)
  }

  sliceResp, err := response.AsFloat16Slice("logits")
  if err != nil {
    log.Fatal(err)
  }
  shape, err := response.GetShape("logits")
  if err != nil {
    log.Fatalf("Failed to extract shape from response: %v", err)
  }

  // Reshape the logits
  reshapedEmbeddings, err := converter.Reshape3D[float64](sliceResp, shape)
  if err != nil {
    log.Fatal("Failed to parse inference response")
  }

  postprocessManager := postprocess.NewPostprocessManager()
  meanPooledEmbeddings, err := postprocessManager.MeanPoolingFloat64Slice3D(reshapedEmbeddings, [][]int64{{1, 1, 1}, {1, 1, 1}})
  if err != nil {
    log.Fatal(err)
  }
  fmt.Println(meanPooledEmbeddings)
}
```

### Handling Different Data Types
The SDK supports various data types such as INT64, BYTES, FP32, etc. Ensure that the data types match those expected by your Triton models.

### Adding Custom Parameters
You can pass custom parameters to inference requests to control model behavior.

```go
customParams := map[string]any{
    "classification": 5,
    "binary_data":    true,
}

response, err := tritonClient.Infer(
    context.Background(),
    "model_name",
    "1",
    inputs,
    outputs,
    nil, nil, nil, nil, nil, nil, nil, nil, customParams,
    &options.InferOptions{},
)
```

### Examples

#### End-to-End Example with Triton Inference Server

This section demonstrates how to set up the Triton Inference Server with the `ty_bert` and `ty_roberta` models from HuggingFace, use the `tokenizer` package to encode text, perform inference, and retrieve the results

```
go get github.com/Trendyol/go-triton-client/examples
cd examples/end2end_inference
docker build -t go-triton-client-end2end-inference .
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 go-triton-client-end2end-inference
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

### Steps to Contribute

#### 1. Fork The Repository

Click the "Fork" button at the top right of the repository page to create a copy in your GitHub account.

#### 2. Clone Your Fork

```
git clone https://github.com/your-username/go-triton-client.git
cd go-triton-client
```

#### 3. Create a Feature Branch

```
git checkout -b feature/your-feature-name
```

#### 4. Make Your Changes

Implement your feature or fix the bug.

#### 5. Run Tests

Ensure all tests pass.
```
go test ./...
```

#### 6. Commit Your Changes

```
git add .
git commit -m "Add feature: your feature description"
```

#### 7. Push to Your Fork

```
git push origin feature/your-feature-name
```

#### 8. Create a Pull Request
