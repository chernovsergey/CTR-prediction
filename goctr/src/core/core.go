package core

import (
	"math"
	"hash/fnv"
	"os"
	"log"
)

func Sigmoid(x, bound float64) float64 {
	return 1.0 / (1.0 + math.Exp(-math.Max(math.Min(x, bound), -bound)))
}

func LogLoss(prob float64, target float64) float64 {
	p := math.Max(math.Min(prob, 1.0 - 10e-15), 10e-15)
	if target == 1 {
		return -math.Log(p)
	} else {
		return -math.Log(1. - p)
	}
}

func StringHash(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}

func CreateLogFile() *os.File {
	f, err := os.OpenFile("log.txt", os.O_WRONLY | os.O_CREATE, 0644)
	if err != nil {
		log.Fatal(err)
	}
	return f
}