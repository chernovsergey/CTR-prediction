package linear_models

import "../core"

type LogisticRegression struct {
	LearnRate float64
	Lambda    float64
	Params    core.Matrix
}

func (model *LogisticRegression) Predict(x core.FeatureVector) float64 {
	wTx := 0.0
	for i := 0; i < len(x); i++ {
		w, ok := model.Params[x[i]]
		if !ok {
			w = 0.0
		}
		wTx += w
	}
	return core.Sigmoid(wTx, 35.0)
}

func (model *LogisticRegression) Update(x core.FeatureVector, p, y float64) {
	g := p - y

	for i := 0; i < len(x); i++ {
		w, ok := model.Params[x[i]]
		if !ok {
			w = 0.0
		}
		w -= model.LearnRate * g + model.Lambda * w
		model.Params[x[i]] = w
	}
}