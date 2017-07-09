package datasets

import (
	"strconv"
	"encoding/csv"
	"../core"
	"os"
	"log"
)

type AvazuSample struct {
	ID       string
	Y        float64
	Date     int
	Features core.FeatureVector
}

func (a AvazuSample) GetFeature() core.FeatureVector {
	return a.Features
}

func (a AvazuSample) GetY() float64 {
	return a.Y
}

type AvazuDataset struct {
	Colnames map[string]int
	PRIME    uint32
	CSV      *csv.Reader
	FileName string
}

func (d *AvazuDataset) Init() {
	file, err := os.Open(d.FileName)
	if err != nil {
		log.Fatal(err)
	}

	csvreader := csv.NewReader(file)
	d.CSV = csvreader

	header, _ := csvreader.Read()
	colnames := make(map[string]int)
	for i, name := range header {
		colnames[name] = i
	}
	d.Colnames = colnames
}

func (d AvazuDataset) NextSample() (core.Sample, bool) {
	row, err := d.CSV.Read()
	if err != nil {
		return AvazuSample{"", 0, 0, nil}, false
	}

	// Read ID
	id := row[d.Colnames["id"]]

	// Read Target
	y := 0.0
	numFeatures := len(row) - 1
	i, click := d.Colnames["click"]
	if click == true {
		if row[i] == "1" {
			y = 1.0
		}
		numFeatures -= 1
	}

	// Read Date
	date, _ := strconv.Atoi(row[d.Colnames["hour"]][4:6])
	date -= 20
	row[d.Colnames["hour"]] = row[d.Colnames["hour"]][6:]

	// Prepare Features
	features := make([]uint32, numFeatures)
	count := 0
	for i := 0; i < len(row); i++ {
		if i != d.Colnames["id"] {
			if click == false || i != d.Colnames["click"] {
				var value = strconv.Itoa(count) + "_" + row[i]
				features[count] = core.StringHash(value) % d.PRIME
				count += 1
			}
		}
	}

	return AvazuSample{id, y, date, features}, true
}
