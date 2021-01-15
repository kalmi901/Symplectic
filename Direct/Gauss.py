import numpy as np

from codecs import decode
import struct


def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]


def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.

        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]


def float_to_bin(value):  # For testing.
    """ Convert float to 64-bit binary string. """
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return '{:064b}'.format(d)
	

def bin_subtract(a, b):
	""" Calculate c = a-b using 1's Complement"""
	n = len(b)
	a = list(a)
	b = list(b)
	c = [0 for i in range(n)]
	for i in range(n):
		a[i] = int(a[i]) 
		b[i] = int(b[i])
	b = bin_complement(b)
	#print(a)
	#print(b)
	#input()
	carry = 0
	for i in reversed(range(n)):
		temp = a[i] + b[i] + carry
		if (temp >= 2):
			carry = 1
			c[i] = temp%2
		else:
			c[i] = temp
			carry = 0
	
	negative = 0
	if (carry == 0):
		negative = 1
		c = bin_complement(c)
	
	if (carry == 1):
		for i in reversed(range(n)):
			temp = c[i] + carry
			#print(c)
			#input()
			if (temp >= 2):
				carry = 1
				c[i] = temp%2
			else:
				c[i] = temp
				carry = 0
			if (carry == 0):
				break
	
	c = ''.join(str(n) for n in c)
	#print(c)
	return c

def bin_extract(a):
	sign = a[0]
	exp = a[1:12]
	man = a[12:]
	
	return sign, exp, man
	
def bin_concat(sign, exp, man):
	return sign + exp + man
	
def bin_complement(x):
	n = len(x)
	for i in range(n):
		if x[i] == 0:
			x[i] = 1
		elif x[i] == 1:
			x[i] = 0
	return x

def bin_leftshift(bit, mask, shift):
	n = len(bit)
	c = [0] * n
	for i in range(0, n):
		c[i] = int(bit[i])
	for i in range(0, n):
		if (bit[i] == "1") and (mask[i] == "1"):
			j = i + shift
			if (j < n):
				#c[i] = 0
				c[j] = 1
	
	c = ''.join(str(n) for n in c)
	return c

# https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
# https://www.sciencedirect.com/science/article/pii/S0743731516301897
# https://apps.dtic.mil/dtic/tr/fulltext/u2/a313755.pdf

# -------------------- Gaussian elimination ---------------------

def ge_solve(A: np.array, b: np.array, pivot = True):
	# Gauss-elimination
	A, b = __upper_triangleMatrix(A, b, pivot)
	x = __backward_substitution(A, b)

	return x
	
	
def __upper_triangleMatrix(A: np.array, b: np.array, pivot = True):
	n = len(A)
	for j in range(n):
		# Search for the maximum in this column (if pivot True - default case)
		#print(A)
		#print(b)
		#input()
		if (pivot == True):
			maxA = 0.0
			jmax = j
			for k in range(j, n):
				absA = abs(A[k, j])
				if (absA > maxA):
					maxA = absA
					jmax = k
			if (j != jmax):
				# Swap the rows
				A[[j, jmax]] = A[[jmax, j]]
				b[[j, jmax]] = b[[jmax, j]]
		#print(A)
		#print(b)
		#input()
		# Make all rows below this one 0 in current column
		for k in range(j+1, n):
			c = -A[k, j] / A[j, j]
			b[k] += c * b[j]
			for i in range(j, n):
				if (j == i):
					A[k, j] = 0
				else:
					A[k, i] += c * A[j, i]
	
	return A, b

# ---------------- Division-Free Gaussian elimination ------------

def dge_solve(A: np.array, b: np.array, pivot = True):
	# Division-Free Gauss-elimination
	#A, b, = __upper_triangleMatrix2(A, b, pivot)
	A, b, = __upper_triangleMatrix3(A, b, pivot)
	#A, b, = __upper_triangleMatrix4(A, b, pivot)
	#print(A)
	x = __backward_substitution(A, b)

	return x
	
	
def __upper_triangleMatrix2(A: np.array, b: np.array, pivot = True):
	n = len(A)
	for j in range(n):
		# Search for the maximum in this column (if pivot True - default case)
		#print(A)
		#input()
		if (pivot == True):
			maxA = 0.0
			jmax = j
			for k in range(j, n):
				absA = abs(A[k, j])
				if (absA > maxA):
					maxA = absA
					jmax = k
			if (j != jmax):
				# Swap the rows
				A[[j, jmax]] = A[[jmax, j]]
				b[[j, jmax]] = b[[jmax, j]]
		#print(A)
		#input()
		# Make all rows below this 0 in current column
		# "Avoid" division
		M = A[j, j]
		#exp = np.log2(M)
		for k in range(j+1, n):
			L = A[k, j]
			b[k] = M * b[k] - L * b[j]
			#b[k] = np.sign(b[k]) * np.power(2, np.log2(abs(b[k]))-exp)
			for i in range(j, n):
				if (j == i):
					A[k, j] = 0
				else:
					A[k, i] = M * A[k, i] - L * A[j, i]
					#A[k, i] = np.sign(A[k, i]) * np.power(2, np.log2(abs(A[k, i]))-exp)

	return A, b
	
	
def __upper_triangleMatrix3(A: np.array, b: np.array, pivot = True):
	n = len(A)	
	for j in range(n):
		# Search for the maximum in this column (if pivot True - default case)
		#print(A)
		#input()
		if (pivot == True):
			maxA = 0.0
			jmax = j
			for k in range(j, n):
				absA = abs(A[k, j])
				if (absA > maxA):
					maxA = absA
					jmax = k
			if (j != jmax):
				# Swap the rows
				A[[j, jmax]] = A[[jmax, j]]
				b[[j, jmax]] = b[[jmax, j]]
		#print(A)
		#input()
		# Make all rows below this 0 in current column
		# "Avoid" division
		M = A[j, j]
		#exp = np.log2(M)
		Mbin = float_to_bin(M)
		_, exp, _ = (bin_extract(Mbin))
		exp = bin_subtract(exp, '01111111111')		# Expl - 1023
		for k in range(j+1, n):
			L = A[k, j]
			b[k] = M * b[k] - L * b[j]
			#b[k] = np.sign(b[k]) * np.power(2, np.log2(abs(b[k]))-exp)
			bbin = float_to_bin(b[k])
			s, e, v = bin_extract(bbin)
			e = bin_subtract(e, exp)
			bbin = bin_concat(s, e, v)
			b[k] = bin_to_float(bbin)
			for i in range(j, n):
				if (j == i):
					A[k, j] = 0
				else:
					A[k, i] = M * A[k, i] - L * A[j, i]
					#A[k, i] = np.sign(A[k, i]) * np.power(2, np.log2(abs(A[k, i]))-exp)
					Abin = float_to_bin(A[k, i])
					s, e, v = bin_extract(Abin)
					e = bin_subtract(e, exp)
					Abin = bin_concat(s, e, v)
					A[k, i] = bin_to_float(Abin)

	return A, b

def __upper_triangleMatrix4(A: np.array, b: np.array, pivot = True):
	n = len(A)
	mask = 9218868437227405312 # 7FF0000000000000		for double precision
	#mask = "0" + bin(mask)[2:]
	for j in range(n):
		# Search for the maximum in this column (if pivot True - default case)
		#print(A)
		#input()
		if (pivot == True):
			maxA = 0.0
			jmax = j
			for k in range(j, n):
				absA = abs(A[k, j])
				if (absA > maxA):
					maxA = absA
					jmax = k
			if (j != jmax):
				# Swap the rows
				A[[j, jmax]] = A[[jmax, j]]
				b[[j, jmax]] = b[[jmax, j]]
		#print(A)
		#input()
		# Make all rows below this 0 in current column
		# "Avoid" division
		M = A[j, j]
		#exp = np.log2(M)
		Mbin = float_to_bin(M)
		_, exp, _ = (bin_extract(Mbin))
		exp = bin_subtract(exp, '01111111111')		# Expl - 1023
		shift = int(exp, 2)							# Left right bit shift parameter
		print(shift)
		for k in range(j+1, n):
			print(k)
			L = A[k, j]
			b[k] = M * b[k] - L * b[j]
			bbin = float_to_bin(b[k])				# binary string of dual number
			tempInt = int(bbin, 2)					# binary string converted to integer number (64 bit)
			tempInt = (tempInt >> shift) & mask | tempInt & ~mask		# bit hack
			tempBin = bin(tempInt)[2:]			# "0b" is removed; due to the conversion n zeros are lost
			bbin = "0" * (64-len(tempBin)) + tempBin	# signature from the original binary; "0"x n missing digits, binary string 
			print(bbin)
			s, e, v = bin_extract(float_to_bin(b[k]))
			e = bin_subtract(e, exp)
			bbin = bin_concat(s, e, v)
			print(bbin)
			input()
			b[k] = bin_to_float(bbin)				# get the number
			for i in range(j, n):
				if (j == i):
					A[k, j] = 0
				else:
					A[k, i] = M * A[k, i] - L * A[j, i]
					Abin = float_to_bin(A[k, i])
					s, e, v = bin_extract(Abin)
					e = bin_subtract(e, exp)
					Abin = bin_concat(s, e, v)
					#Abin = bin_leftshift(Abin, mask, shift)
					A[k, i] = bin_to_float(Abin)

	return A, b
# ------------------------- Common --------------------------------
	
def __backward_substitution(U: np.array, b: np.array):
	n = len(U)					# Size of the matrix
	x = np.zeros([n])			# Empty array to store the results
	for i in reversed(range(n)):
		s = sum(U[i, j] * x[j] for j in range(i, n))
		x[i] = (b[i] - s) / U[i,i]
		
	return x