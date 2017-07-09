package benchmarks

import (
	"time"
	"log"
	"../src/core"
)

type BenchMark struct {
	Data            core.Dataset
	Model           core.Model
	NumEpoch        int
	EvalStep        int
	VerboseEvalStep int
}

func (b BenchMark) Epoch() float64 {
	line := 1
	evalSize := 0.0
	loss := 0.0
	b.Data.Init()

	for {
		sample, ok := b.Data.NextSample()
		features := sample.GetFeature()
		if !ok {
			break
		}

		p := b.Model.Predict(features)
		if line % b.EvalStep == 0 {
			evalSize += 1
			loss += core.LogLoss(p, sample.GetY())
		}

		if line % b.VerboseEvalStep == 0 {
			log.Printf("Line: %d  Loss: %f", line, loss / evalSize)
		}
		b.Model.Update(features, p, sample.GetY())
		line += 1
	}

	return loss / evalSize

}

func (b BenchMark) Start() {
	for ep := 0; ep < b.NumEpoch; ep++ {
		t0 := time.Now()
		loss := b.Epoch()
		t1 := time.Since(t0)
		log.Printf("Epoch %d took %s logloss %f", ep + 1, t1, loss)
	}
}
