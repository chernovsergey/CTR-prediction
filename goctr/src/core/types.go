package core

type Matrix map[uint32]float64
type FeatureVector []uint32

type Model interface {
	Predict(FeatureVector) float64
	Update(FeatureVector, float64, float64)
}

type Dataset interface {
	Init()
	NextSample() (Sample, bool)
}

type Sample interface {
	GetFeature() FeatureVector
	GetY() float64
}

type Benchmark interface {
	Start()
}