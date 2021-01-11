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
	
num = 126.52234
maskInt = 9218868437227405312 # 7FF0000000000000		for double precision
binaryMask = "0" + bin(maskInt)[2:]
print(binaryMask)
binaryNum = float_to_bin(num)
print(binaryNum)
# Bit hack on int variables
binaryInt = int(binaryNum, 2)
binaryInt = (binaryInt >> 5) & maskInt | binaryInt & ~maskInt
temp = bin(binaryInt)[2:]
binaryRes = "0" * (64-len(temp)) + temp
print(binaryRes)


#s, e, m = bin_extract(binary)


"""
a = 64
mask = 9
print(bin(a))
a = (a >> 5) & mask
print(a)
print(bin(a))
print(bin(mask))
"""

#print(s)
#print(e)
#print(int(e,2)-1023)
