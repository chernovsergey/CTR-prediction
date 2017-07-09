package benchmarks

import (
	"testing"
	"../src/datasets"
	"../src/linear_models"
)

func BenchmarkFTRL(*testing.B) {

	train := "../../ctrdata/avazu/train.csv"
	epoch := 5
	var verboseEval int = 10000
	holdout := 30
	var prime uint32 = 15485863


	dataset := datasets.AvazuDataset{FileName: train, PRIME:prime}

	clf := linear_models.LogisticRegression{
		LearnRate: 0.02,
		Lambda:0.0001,
		Params: make(map[uint32]float64),
	}

	//clf := linear_models.FTRL{
	//	Alpha: 0.05,
	//	Beta: 1.1,
	//	L1: 1.1,
	//	L2:1.1,
	//	N: make(map[uint32]float64),
	//	Z: make(map[uint32]float64),
	//	W: make(map[uint32]float64),
	//}

	benchmark := BenchMark{
		Data:&dataset, Model:&clf,
		NumEpoch:epoch, VerboseEvalStep:verboseEval, EvalStep:holdout,
	}

	benchmark.Start()
}