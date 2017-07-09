package linear_models

import (
	"math"
	"../core"
)

type FTRL struct {
	Alpha, Beta, L1, L2 float64
	N                   core.Matrix // squared sum of past gradients
	Z                   core.Matrix // coefficients / weights
	W                   core.Matrix // tmp coefficients / weights
}

func (model *FTRL) Predict(x core.FeatureVector) float64 {
	wTx := 0.0
	for i := 0; i < len(x); i++ {

		z, ok := model.Z[x[i]]
		if ok == false {
			model.Z[x[i]] = 0
			model.N[x[i]] = 0
			model.W[x[i]] = 0
			z = 0
		}

		sign := 1.0
		if z < 0 {
			sign = -1.0
		}

		if sign * z <= model.L1 {
			model.W[x[i]] = 0.
		} else {
			model.W[x[i]] = (sign * model.L1 - z) / ((model.Beta + math.Sqrt(model.N[x[i]])) / model.Alpha + model.L2)
		}

		wTx += model.W[x[i]]
	}

	return core.Sigmoid(wTx, 35.0)
}

func (m *FTRL) Update(x core.FeatureVector, p, y float64) {

	// Gradient under logloss
	g := p - y

	// update z and n
	for i := 0; i < len(x); i++ {
		sigma := (math.Sqrt(m.N[x[i]] + g * g) - math.Sqrt(m.N[x[i]])) / m.Alpha
		m.Z[x[i]] += g - sigma * m.W[x[i]]
		m.N[x[i]] += g * g
	}
}

