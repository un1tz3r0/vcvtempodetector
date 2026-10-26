#pragma once
#include <cmath>
#include <complex>
#include <vector>

namespace beat {

/** Real-input FFT of a power-of-two size n: an n/2-point complex FFT of the even/odd samples,
then the usual split into the n/2 + 1 non-negative frequency bins. */
class RealFft {
	using cplx = std::complex<double>;
	int n, m;
	std::vector<int> rev;
	std::vector<cplx> twiddle, split, z;

public:
	explicit RealFft(int n) : n(n), m(n / 2), rev(m), twiddle(m / 2), split(m + 1), z(m) {
		int bits = 0;
		while ((1 << bits) < m)
			bits++;
		for (int i = 0; i < m; i++)
			for (int b = 0; b < bits; b++)
				rev[i] |= ((i >> b) & 1) << (bits - 1 - b);
		for (int k = 0; k < m / 2; k++)
			twiddle[k] = std::polar(1.0, -2 * M_PI * k / m);
		for (int k = 0; k <= m; k++)
			split[k] = std::polar(1.0, -2 * M_PI * k / n);
	}

	int size() const { return n; }

	/** x has n samples; out receives n/2 + 1 bins, unnormalised (as numpy.fft.rfft). */
	void forward(const double* x, cplx* out) {
		for (int i = 0; i < m; i++)
			z[rev[i]] = {x[2 * i], x[2 * i + 1]};
		for (int len = 2; len <= m; len <<= 1) {
			int stride = m / len;
			for (int i = 0; i < m; i += len)
				for (int k = 0; k < len / 2; k++) {
					cplx &a = z[i + k], &b = z[i + k + len / 2], w = twiddle[k * stride];
					double br = b.real() * w.real() - b.imag() * w.imag(), bi = b.real() * w.imag() + b.imag() * w.real();
					b = {a.real() - br, a.imag() - bi};
					a = {a.real() + br, a.imag() + bi};
				}
		}
		for (int k = 0; k <= m; k++) {
			cplx zk = z[k % m], zc = std::conj(z[(m - k) % m]), w = split[k];
			double er = 0.5 * (zk.real() + zc.real()), ei = 0.5 * (zk.imag() + zc.imag());
			double orr = 0.5 * (zk.imag() - zc.imag()), oi = -0.5 * (zk.real() - zc.real());  // -i/2 (zk - zc)
			out[k] = {er + w.real() * orr - w.imag() * oi, ei + w.real() * oi + w.imag() * orr};
		}
	}
};

} // namespace beat
