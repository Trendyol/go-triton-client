// Code generated by MockGen. DO NOT EDIT.
// Source: base/grpc_client.go

// Package mocks is a generated GoMock package.
package mocks

import (
	reflect "reflect"

	gomock "github.com/golang/mock/gomock"
	grpc "google.golang.org/grpc"
)

// MockGrpcClient is a mock of GrpcClient interface.
type MockGrpcClient struct {
	ctrl     *gomock.Controller
	recorder *MockGrpcClientMockRecorder
}

// MockGrpcClientMockRecorder is the mock recorder for MockGrpcClient.
type MockGrpcClientMockRecorder struct {
	mock *MockGrpcClient
}

// NewMockGrpcClient creates a new mock instance.
func NewMockGrpcClient(ctrl *gomock.Controller) *MockGrpcClient {
	mock := &MockGrpcClient{ctrl: ctrl}
	mock.recorder = &MockGrpcClientMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockGrpcClient) EXPECT() *MockGrpcClientMockRecorder {
	return m.recorder
}

// GetConnection mocks base method.
func (m *MockGrpcClient) GetConnection() *grpc.ClientConn {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetConnection")
	ret0, _ := ret[0].(*grpc.ClientConn)
	return ret0
}

// GetConnection indicates an expected call of GetConnection.
func (mr *MockGrpcClientMockRecorder) GetConnection() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetConnection", reflect.TypeOf((*MockGrpcClient)(nil).GetConnection))
}