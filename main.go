package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/gonum/matrix/mat64"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func main() {
	m, n := 1000, 10
	r := make([][]float64, m)
	for i := range r {
		r[i] = make([]float64, n)
		for j := range r[i] {
			if rand.Int()%2 == 0 {
				r[i][j] = float64(rand.Intn(5) + 1)
			}
		}
	}

	// should be strictly less than min(columns, rows)
	features := 2
	// regularization parameter, should be small-ish
	lambda := 0.1
	// how many times to run the solver
	iterations := 10000

	start := time.Now()
	p, q := factorize(r, features, lambda, iterations)
	end := time.Now()

	// the original values are copied from here:
	// http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/#basic-ideas
	fmt.Println("original")
	fmt.Printf("% v\n", mat64.Formatted(toDense(r)))

	// the recommendations are similar in value to the result from the same source:
	// http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/#implementation-in-python
	fmt.Println("\nrecommendations")
	out := mat64.NewDense(len(p[0]), len(q[0]), nil)
	out.Mul(toDense(p).T(), toDense(q))
	fmt.Printf("% .2f\n", mat64.Formatted(out, mat64.Squeeze()))

	// the feature matrices can be saved to recalculate recommendations
	// but i think the recommendations matrix takes less space overall, so not sure if necessary
	//fmt.Println("\nrow features")
	//fmt.Println(mat64.Formatted(toDense(p).T()))
	//fmt.Println("\ncolumn features")
	//fmt.Println(mat64.Formatted(toDense(q).T()))

	// time
	fmt.Printf("\ntime: %dms\n", end.Sub(start)/time.Millisecond)
}

func factorize(r [][]float64, k int, lambda float64, count int) (p, q [][]float64) {
	// initialize q to random values as a starting point
	q = make([][]float64, k)
	for i := range q {
		q[i] = make([]float64, len(r[0]))
		for j := range q[i] {
			q[i][j] = rand.Float64()
		}
	}

	p = make([][]float64, k)
	for i := range p {
		p[i] = make([]float64, len(r))
	}

	for i := 0; i < count; i++ {
		solveP(p, q, r)
		solveQ(p, q, r)
	}

	return
}

func solveP(p, q, r [][]float64) {
	λ := 0.1
	k := len(p) // or len(q)

	// each loop can be calculated independently
	for pj := range p[0] {
		// Q^T x w_i x Q + λI
		// k by k
		f := make([]float64, k*k)
		for i := 0; i < k; i++ {
			for j := 0; j < k; j++ {
				var t float64
				for a := range q[i] {
					if r[pj][a] > 0 {
						t += q[i][a] * q[j][a]
					}
				}
				// regularization
				if i == j {
					t += λ
				}
				// write into matrix
				f[j*k+i] = t
			}
		}

		// Q^T x w_i x r_i
		// k by 1
		s := make([]float64, k)
		for i := 0; i < k; i++ {
			var t float64
			for a := range q[i] {
				if r[pj][a] > 0 {
					t += q[i][a] * r[pj][a]
				}
			}
			s[i] = t
		}

		v := make([]float64, k)
		mat64.NewDense(k, 1, v).Solve(
			mat64.NewDense(k, k, f),
			mat64.NewDense(k, 1, s),
		)
		for i := 0; i < k; i++ {
			p[i][pj] = v[i]
		}
	}
}

func solvePj(p, q [][]float64, rj []float64, pj int) {
	// Q^T x w_i x Q + λI
	// k by k
	f := make([]float64, k*k)
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			var t float64
			for a := range q[i] {
				if ri[a] > 0 {
					t += q[i][a] * q[j][a]
				}
			}
			// regularization
			if i == j {
				t += λ
			}
			// write into matrix
			f[j*k+i] = t
		}
	}

	// Q^T x w_i x r_i
	// k by 1
	s := make([]float64, k)
	for i := 0; i < k; i++ {
		var t float64
		for a := range q[i] {
			if rj[a] > 0 {
				t += q[i][a] * rj[a]
			}
		}
		s[i] = t
	}

	v := make([]float64, k)
	mat64.NewDense(k, 1, v).Solve(
		mat64.NewDense(k, k, f),
		mat64.NewDense(k, 1, s),
	)
	for i := 0; i < k; i++ {
		p[i][pj] = v[i]
	}
}

func solveQ(p, q, r [][]float64) {
	λ := 0.1
	k := len(p) // or len(q)

	// each loop can be calculated independently
	for qj := range q[0] {
		// P^T x w_j x P + λI
		// k by k
		f := make([]float64, k*k)
		for i := 0; i < k; i++ {
			for j := 0; j < k; j++ {
				var t float64
				for a := range p[i] {
					if r[a][qj] > 0 {
						t += p[i][a] * p[j][a]
					}
				}
				// regularization
				if i == j {
					t += λ
				}
				// write into matrix
				f[j*k+i] = t
			}
		}

		// P^T x w_j x r_j
		// k by 1
		s := make([]float64, k)
		for i := 0; i < k; i++ {
			var t float64
			for a := range p[i] {
				if r[a][qj] > 0 {
					t += p[i][a] * r[a][qj]
				}
			}
			s[i] = t
		}

		v := make([]float64, k)
		mat64.NewDense(k, 1, v).Solve(
			mat64.NewDense(k, k, f),
			mat64.NewDense(k, 1, s),
		)
		for i := 0; i < k; i++ {
			q[i][qj] = v[i]
		}
	}
}

func toDense(m [][]float64) *mat64.Dense {
	i, j := len(m), len(m[0])
	out := make([]float64, i*j)
	for k := range m {
		copy(out[k*len(m[k]):], m[k])
	}
	return mat64.NewDense(i, j, out)
}
