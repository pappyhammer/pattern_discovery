
'''

Core functions for fitting the CILVA model.

Author: Marcus A. Triplett. (2019). University of Queensland, Australia.


'''

import scipy as sp
import scipy.signal as sg
import numpy as np
import time


'''

	Log-joint probability densities of the observed data and latent variables.

'''

def log_joint(f, s, kernel, N, T, K, L, gamma, sigma, alpha, beta, w, b, x):
	lambda_ = w @ s + b @ x
	conv = np.array([np.convolve(kernel, lambda_[n, :])[:T] for n in range(N)])
	err = (f - alpha[:, None] * conv - beta[:, None]) ** 2
	noise_coefs = -1/(2 * sigma ** 2)
	sse = np.sum(noise_coefs[:, None] * err) - np.sum(x)/gamma
	return -sse # Negative for minimisation

def log_joint_flat_static(static_params, args):
	f, s, kernel, N, T, K, L, gamma, sigma, x = args
	lens = [N, K, L]
	alpha, beta, w, b = unflatten_static_params(static_params, lens)
	return log_joint(f, s, kernel, N, T, K, L, gamma, sigma, alpha, beta, w, b, x)

def log_joint_flat_latent(latent_vars, args):
	f, s, kernel, N, T, K, L, gamma, sigma, alpha, beta, w, b = args
	lens = [L, T]
	x = unflatten_latent_vars(latent_vars, lens)
	return log_joint(f, s, kernel, N, T, K, L, gamma, sigma, alpha, beta, w, b, x)


'''

	Parameter reshaping.

'''

def flatten_static_params(alpha, beta, w, b, lens):
	N, K, L = lens
	arr_len = N + N + (N * K) + (N * L)
	arr = np.zeros(arr_len); loc = 0
	arr[loc:loc + N] = alpha[:]; loc += N
	arr[loc:loc + N] = beta[:]; loc += N
	arr[loc:loc + N * K] = w.ravel(); loc += N * K
	arr[loc:loc + N * L] = b.ravel()
	return arr

def unflatten_static_params(arr, lens):
	N, K, L = lens
	loc = 0
	alpha = arr[loc:loc + N]; loc += N
	beta = arr[loc:loc + N]; loc += N
	w = np.reshape(arr[loc:loc + N * K], [N, K]); loc += N * K
	b = np.reshape(arr[loc:loc + N * L], [N, L])
	return [alpha, beta, w, b]

def flatten_latent_vars(latent_vars):
	return latent_vars.ravel()

def unflatten_latent_vars(arr, lens):
	L, T = lens
	return np.reshape(arr, [L, T])

'''

	Gradients of the log-joint density function.

'''

def grad_static_params(f, s, kernel, N, T, K, L, gamma, sigma, alpha, beta, w, b, x):
	sigma_sq = sigma ** 2
	lambda_ = w @ s + b @ x

	alpha_div_sigma_sq = np.divide(alpha, sigma_sq)
	alpha_div_sigma_sq_diag = np.diag(alpha_div_sigma_sq)

	conv = np.array([np.convolve(kernel, lambda_[n, :])[:T] for n in range(N)])
	conv_with_stim = np.array([np.convolve(kernel, s[k, :])[:T] for k in range(K)])
	conv_with_x = np.array([np.convolve(kernel, x[l, :])[:T] for l in range(L)])

	err = f - alpha[:, None] * conv - beta[:, None]
	prod = np.multiply(err, conv)

	grad_alpha = np.multiply(alpha_div_sigma_sq, np.sum(prod, 1))
	grad_beta = np.multiply(1/sigma_sq, np.sum(err, 1))
	grad_w = alpha_div_sigma_sq_diag @ (err @ conv_with_stim.T)
	grad_b = alpha_div_sigma_sq_diag @ (err @ conv_with_x.T)

	lens = [N, K, L]

	# Negative gradients for minimisation
	return flatten_static_params(-grad_alpha, -grad_beta, -grad_w, -grad_b, lens)

def grad_latent_vars(f, s, kernel, N, T, K, L, gamma, sigma, alpha, beta, w, b, x):
	sigma_sq = sigma ** 2
	lambda_ = w @ s + b @ x

	alpha_div_sigma_sq = np.divide(alpha, sigma_sq)

	conv = np.array([np.convolve(kernel, lambda_[n, :])[:T] for n in range(N)])
	err = f - alpha[:, None] * conv - beta[:, None]

	gamma_regulariser = 1/gamma * np.ones((L, ))
	grad_x = np.zeros((L, T))
	for t in range(T):
		grad_x[:, t] = (b.T @ np.multiply(alpha_div_sigma_sq, (err[:, t:T] @ kernel[:T - t])) 
			- gamma_regulariser)
		
	return flatten_latent_vars(-grad_x)

def flat_grad_static_params(params, args):
	f, s, kernel, N, T, K, L, gamma, sigma, x = args
	lens = [N, K, L]
	alpha, beta, w, b = unflatten_static_params(params, lens)
	return grad_static_params(f, s, kernel, N, T, K, L, gamma, sigma, alpha, beta, w, b, x)

def flat_grad_latent_vars(params, args):
	f, s, kernel, N, T, K, L, gamma, sigma, alpha, beta, w, b = args
	lens = [L, T]
	x = unflatten_latent_vars(params, lens)
	return grad_latent_vars(f, s, kernel, N, T, K, L, gamma, sigma, alpha, beta, w, b, x)

'''

	Primary model-fitting procedure.

'''

def alternating_minimisation(initial_params, args, static_bounds, latent_bounds, num_iters,
	iters_per_altern):
	
	'''

		Alternating optimisation of the MAP estimator.

	'''

	print('\nBeginning MAP estimation.\n')

	alpha, beta, w, b, x = initial_params
	f, s, kernel, N, T, K, L, gamma, sigma = args

	xlens = [L, T]
	plens = [N, K, L]

	for it in range(num_iters):
		# Update latent vars
		print('Alternation %i/%i: latent variables\n'%(it + 1, num_iters))
		tstart = time.time()
		args = [f, s, kernel, N, T, K, L, gamma, sigma, alpha, beta, w, b]
		results_latents = sp.optimize.minimize(log_joint_flat_latent,
			flatten_latent_vars(x),
			args = args,
			method = 'L-BFGS-B',
			jac = flat_grad_latent_vars,
			bounds = latent_bounds,
			options = {'disp': True, 'maxiter': iters_per_altern}).x
		tstop = time.time()
		print('Alternation time %.2fs\n'%(tstop - tstart))
		x = unflatten_latent_vars(results_latents, xlens)

		# Update static params
		print('Alternation %i/%i: static parameters\n'%(it + 1, num_iters))
		tstart = time.time()
		args = [f, s, kernel, N, T, K, L, gamma, sigma, x]
		results_params = sp.optimize.minimize(log_joint_flat_static,
			flatten_static_params(alpha, beta, w, b, plens),
			args = args,
			method = 'L-BFGS-B',
			jac = flat_grad_static_params,
			bounds = static_bounds,
			options = {'disp': True, 'maxiter': iters_per_altern}).x
		tstop = time.time()
		print('Alternation time %.2fs\n'%(tstop - tstart))
		alpha, beta, w, b = unflatten_static_params(results_params, plens)

	print('Optimisation complete.', end=" ")
	return alpha, beta, w, b, x

def estimate_latents(initial_params, args, bounds, num_iters):
	'''

		Optimisation of the MAP estimator for latent variables only.

	'''

	print('\nBeginning MAP estimation of latent variables.\n')

	x = initial_params
	L, T = x.shape

	results_latents = sp.optimize.minimize(log_joint_flat_latent,
		flatten_latent_vars(x),
		args = args,
		method = 'L-BFGS-B',
		jac = flat_grad_latent_vars,
		bounds = bounds,
		options = {'disp': True, 'maxiter': num_iters}).x

	x = unflatten_latent_vars(results_latents, [L, T])

	return x

'''

	Initialisation funcs.

'''

def load_data(data, convert=False):
	f = np.loadtxt(data + '.ca2')
	s = np.loadtxt(data + '.stim')
	if convert: s = convert_stim(s)
	return [f, s]


def init_filters(f, s, kernel, N, T, K):
	'''

		Initialise stimulus filters w via non-negative least squares regression of
		fluorescence traces onto a basis of stimulus regressors.

	'''
	stim_regressors = np.zeros((T, K))
	for k in range(K):
		stim_regressors[:, k] = np.convolve(kernel, s[k, :])[:T]
		# fig, ax = plt.subplots(nrows=1, ncols=1,
		# 					   gridspec_kw={'height_ratios': [1]},
		# 					   figsize=(15, 2))
		# ax.plot(stim_regressors[:, k])
		# fig.savefig(f'/media/julien/Not_today/hne_not_today/results_hne/test_stim.pdf',
		# 			format=f"pdf",
		# 			facecolor=fig.get_facecolor())
		# plt.show()
		# raise Exception("TOTO")
	w_nnls = np.zeros((N, K))
	for n in range(N):
		w_nnls[n, :] = sp.optimize.nnls(stim_regressors, f[n, :])[0]

	return w_nnls


def estimate_noise_sdevs(f, N, T, imrate):
	'''
		
		Estimate standard deviations of the imaging noise. Uses method suggested
		in Pnevmatikakis et al. (2016).

	'''

	sdevs = np.zeros(N)

	# Find frequency in periodogram closest to the desired frequency
	nearest = lambda freqs, val: min(freqs, key=lambda x: np.abs(x - val))
	sample_freqs, psd = sg.periodogram(f[0, :], imrate, scaling='density')
	start_freq = np.where(sample_freqs == nearest(sample_freqs, imrate/4))[0][0]
	end_freq = np.where(sample_freqs == nearest(sample_freqs, imrate/2))[0][0]
	sdevs[0] = np.sqrt(np.mean(psd[start_freq:end_freq]))

	for n in range(1, N):
		sample_freqs, psd = sg.periodogram(f[n, :], imrate, scaling='density')
		sdevs[n] = np.sqrt(np.mean(psd[start_freq:end_freq]))

	return sdevs


def convert_stim(stim_vec):
	'''

		Convert from a 1d to 2d stimulus representation.

	'''

	duration = stim_vec.shape[0]
	K = int(np.max(stim_vec))
	s = np.zeros((K, duration))
	for t in range(duration):
		stim_indx = int(stim_vec[t])
		if stim_indx != 0:        
			s[stim_indx - 1, t] = 1
	return s

def calcium_kernel(tau_r, tau_d, T):
	return np.exp(-np.arange(T)/tau_d) - np.exp(-np.arange(T)/tau_r)


def identify_params(alpha, beta, w, b, x, sigma, param_identification_args):
	'''
		
		Parameter identification and rescaling follow model-fit.

	'''

	N, L, s = param_identification_args

	xnorms = np.array([np.linalg.norm(x[l, :]) for l in range(L)])
	xorder = np.argsort(xnorms)[::-1]
	x = x[xorder, :]
	b = b[:, xorder]
	xnorms = xnorms[xorder]
	for l in range(L):
		x[l, :] /= xnorms[l]
		b[:, l] *= xnorms[l]

	lambda_ = w @ s + b @ x
	lambda_norms = np.array([np.linalg.norm(lambda_[n, :]) for n in range(N)])
	alpha *= lambda_norms
	w /= lambda_norms[:, None]
	b /= lambda_norms[:, None]

	return alpha, beta, w, b, x, sigma


def fit_kernel_time_constants(f, s, N, T, K, eta, num_iters, return_errs=False):
	'''
		Code for learning rise and decay calcium time constants.
		This is a heuristic method for use if the time constants are not known
		to any degree of accuracy. To ensure that the estimated time constants
		are sensible for your data, I suggest using them to generate a calcium
		kernel, convolving this with the stimulus times, and checking that they
		line up with the observed calcium transients. It may require some adjustment
		of the penalty term (potentially by multiple orders of magnitude) before
		suitable time constants are found.
		To use, call the function with the following inputs
			f:
				Matrix of fluorescence levels (of size NxT).
			s:
				Matrix of stimulus times (of size KxT).
			N:
				Number of neurons.
			T:
				Total trial duration.
			eta:
				Penalty on the kernel time constants, to help counteract spontaneous events
				biasing the constants towards larger values.
			num_iters:
				Number of times to repeat the estimation. Convergence typically occurs
				within 3-5 iterations.
			return_errs: bool
				Optional argument. If true, returns a vector of errors to monitor convergence.

	'''

	bounds = [(1e-1, None)] * 2
	tau_r = 0.5  # initial guess
	delta_tau = 1
	errs = np.zeros(num_iters)

	for it in range(num_iters):
		kernel = calcium_kernel(tau_r, delta_tau + tau_r, T)
		beta = fit_regressors(f, s, kernel, N, T, K)
		params = [tau_r, delta_tau]
		args = [f, s, N, T, K, beta, eta]
		tau_r, delta_tau = sp.optimize.minimize(reconstruction_error_tau,
												params,
												args=args,
												method='L-BFGS-B',
												jac=grad_tau,
												bounds=bounds,
												options={'disp': True}).x
		errs[it] = reconstruction_error(f, s, N, T, K, tau_r, delta_tau, beta, eta)

	params = [tau_r, tau_r + delta_tau]
	if return_errs:
		params += [errs]

	return params

def fit_regressors(f, s, kernel, N, T, K):
	'''
		Initialise stimulus filters w via non-negative least squares regression of
		fluorescence traces onto a basis of stimulus regressors.
	'''
	stim_regressors = np.zeros((T, K))
	for k in range(K):
		stim_regressors[:, k] = np.convolve(kernel, s[k, :])[:T]

	w_nnls = np.zeros((N, K))
	for n in range(N):
		w_nnls[n, :] = sp.optimize.nnls(stim_regressors, f[n, :])[0]

	return w_nnls

def grad_tau(params, args):
	tau_r, delta_tau = params
	f, s, N, T, K, beta, eta = args
	Tvec = np.arange(T)
	tau_d = tau_r + delta_tau

	kernel = calcium_kernel(tau_r, tau_d, T)
	stim = beta @ s
	kconv = np.array([np.convolve(kernel, stim[n, :])[:T] for n in range(N)])

	dk_dtau_r = -Tvec / tau_d**2 * np.exp(-Tvec/tau_d) + Tvec / tau_r**2 * np.exp(-Tvec/tau_r)
	dk_ddelta_tau = -Tvec / tau_d**2 * np.exp(-Tvec/tau_d)

	dk_dtau_r_conv = np.array([np.convolve(dk_dtau_r, stim[n, :])[:T] for n in range(N)])
	dk_ddelta_tau_conv = np.array([np.convolve(dk_ddelta_tau, stim[n, :])[:T] for n in range(N)])

	grad_tau_r = np.sum((f - kconv) * dk_dtau_r_conv) + eta
	grad_delta_tau = np.sum((f - kconv) * dk_ddelta_tau_conv) + eta

	return np.array([grad_tau_r, grad_delta_tau])

def reconstruction_error(f, s, N, T, K, tau_r, delta_tau, beta, eta):
	kernel = calcium_kernel(tau_r, tau_r + delta_tau, T)
	conv_with_stim = np.array([np.convolve(kernel, s[k, :])[:T] for k in range(K)])
	err = 1/2 * np.sum((f - beta @ conv_with_stim) ** 2) + eta * (tau_r + delta_tau)
	return err

def reconstruction_error_tau(params, args):
	tau_r, delta_tau = params
	f, s, N, T, K, beta, eta = args
	return reconstruction_error(f, s, N, T, K, tau_r, delta_tau, beta, eta)