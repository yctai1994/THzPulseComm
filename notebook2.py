#!/usr/bin/env python
# coding: utf-8

# # Package & function

# In[137]:


import numpy as np
import matplotlib.pyplot as plt

twopi = 2 * np.pi

def plot(
        x, y, xlabel='None', ylabel='None', 
        title='None'):
    """Plot x, y diagram"""
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")
    plt.xlabel(xlabel, fontsize = 18)
    plt.ylabel(ylabel, fontsize = 18)
    plt.title(title, fontsize = 20)
    plt.show()

def axis(
        N, dt):
    """Make time and frequency axis"""
    t = np.linspace(- 0.5 * (N-1) *dt, 0.5 * (N-1) * dt, N)
    freq = np.fft.fftfreq(N , d=dt)
    return t, freq    

def RT(t):
    """Transmitting antenna response function"""
    tauC = 0.3
    tauT = 0.6
    t_relu = np.maximum(0.0, t)
    u = np.heaviside(t, 1)
    RT_t = u * (1 - np.exp(- t_relu/tauC)) * np.exp(- t_relu/tauT)
    fft = np.fft.fft(RT_t)
    return RT_t, fft

def RR(t):
    """Receiving antenna response function"""
    tauR = 0.6
    t_relu = np.maximum(0.0, t)
    u = np.heaviside(t, 1)
    RR_t = u * np.exp(- t_relu/tauR)
    fft = np.fft.fft(RR_t)
    return RR_t, fft
    
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

class Gaussian:
    def __init__(self, FWHM, mu, GDD, f, t, freq):
        self.sigma = FWHM / (2*np.sqrt(2*np.log(2)))
        self.mu = mu
        self.GDD = GDD
        self.w0 = twopi * f
        self.t = t
        self.w = twopi * freq
    def normal(
        self):
        normal = (1 / (self.sigma * np.sqrt(twopi))) * np.exp(- 0.5 * (1 / self.sigma ** 2) * ((self.t - self.mu)** 2))
        return normal
    def E_t(
            self):
        E_t =  (1 / (self.sigma * np.sqrt(twopi))) * np.exp(- 0.5 * (1 / self.sigma ** 2) * ((self.t - self.mu)** 2)) * np.exp(1j * (self.w0 * self.t + self.GDD * self.t**2))
        return E_t
    def E_w(
            self):
        a = 1 / (2 * self.sigma ** 2)
        c = a**2 + self.GDD**2
        E_w = (1 / (np.sqrt(2) * self.sigma)) * (1 / np.sqrt(a - 1j * self.GDD)) * np.exp(-a * (self.w0 - self.w)**2 / (4*c)) * np.exp(-1j * self.GDD * (self.w0 - self.w)**2 / (4*c))
        return E_w
    def I_t(
            self):
        E_t =  (1 / (self.sigma * np.sqrt(twopi))) * np.exp(- 0.5 * (1 / self.sigma ** 2) * ((self.t - self.mu)** 2)) * np.exp(1j * (self.w0 * self.t + self.GDD * self.t**2))
        I_t = abs(E_t) **2
        return I_t
    def I_w(
            self):
        a = 1 / (2 * self.sigma ** 2)
        c = a**2 + self.GDD**2
        E_w = (1 / (np.sqrt(2) * self.sigma)) * (1 / np.sqrt(a - 1j * self.GDD)) * np.exp(-a * (self.w0 - self.w)**2 / (4*c)) * np.exp(-1j * self.GDD * (self.w0 - self.w)**2 / (4*c))
        I_w = abs(E_w) **2
        return I_w      


# # New Gaussian Class Test

# In[48]:


N = 513
dt = 0.5

t, freq = axis(N, dt)
        
G = Gaussian(10, 0, 0, 0.5, t, freq)

E_t = G.E_t()
E_t_fft = np.fft.fft(E_t)
E_w = G.E_w()
E_w_ifft = np.fft.ifft(E_w)
I_w = G.I_w()

plot(t, E_t, 'Delay Time', 'Amplitude', 'Chirped pulse')
plot(np.fft.fftshift(freq), np.fft.fftshift(E_w), 'Frequency', 'Amplitude', 'E(w)' )
plot(t, np.fft.ifftshift(E_w_ifft), 'Delay Time', 'Amplitude', 'E(t) from frequency domain')
plot(np.fft.fftshift(freq), np.fft.fftshift(E_t_fft), 'Frequency', 'Amplitude', 'Spectrum from time domain')


# ### F{N(t) * RT_t(t)}, N(t) and RT_t(t) with N dots.
# ### F{N(t)} x F{RT_t(t)}, N(t) and RT_t(t) with 2N - 1 dots.

# In[4]:


N = 129
dt = 1

M = 2*N - 1
dT = 0.5 * dt

t, freq = axis(N, dt)
t1, freq1 = axis(M, dT)
        
G = Gaussian(10, 0, 0, 0.5, t, freq)
G1 = Gaussian(10, 0, 0, 0.5, t1, freq1)

Intensity = G.I_t()
Intensity1 = G1.I_t()

RT_t, RT_t_fft = RT(t)
RT_t1, RT_t_fft1 = RT(t1)

"""F{I(t) * RT_t(t)}"""
response3 = np.convolve(Intensity, RT_t, 'full')
response3_fft = np.fft.fft(response3)

""" F{I(t)} x F{RT_t(t)}""" #目前是對的
Intensity1_fft = np.fft.fft(Intensity1)
response4_fft = Intensity1_fft * RT_t_fft1

t_con= np.linspace(-0.5 * (N - 1) * dt, 1.5 * (N - 1) * dt, N*2 -1)
freq_con = np.fft.fftfreq(N*2 - 1 , d=dt)

#plt.plot(np.fft.fftshift(freq_con), np.fft.fftshift(response_fft), label='F{N(t) * RT(t)}')
#plt.plot(np.fft.fftshift(freq1), np.fft.fftshift(response2_fft), label='F{N(t)} x F{RT(t)}')
plt.plot(np.fft.fftshift(freq_con), np.fft.fftshift(response3_fft).imag, label='F{I(t) * RT(t)}')
plt.plot(np.fft.fftshift(freq1), np.fft.fftshift(response4_fft).imag, label='F{I(t)} x F{RT(t)}')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.legend()
plt.show()


# # Readfile

# In[132]:


paper_t = []
paper_amp = []

with open('paper3.txt') as f:
    for l in f.readlines():
        s = l.split('\t')
        paper_t.append(float(s[0]))
        ss = s[1].split('\n')
        paper_amp.append(float(ss[0]))


# # Terahertz system response with single-pulse optial excitation. 

# In[140]:


N = 257
dt = 0.05

t, freq = axis(N, dt)

zt = 1j * freq
zr = 1j * freq

g = Gaussian(FWHM, 0, 0, 1.5, t, freq)


i_t = g.normal() #g.I_t()
i_t_fft = np.fft.fft(i_t)

RT_t, RT_w = RT(t)
RR_t, RR_w = RR(t)

jt_w = i_t_fft * RT_w

radiation_field = zt * jt_w

vr_w = zr * radiation_field * 1

q_w = zt * zr * 1 * RT_w * np.conj(RR_w) * i_t_fft * np.conj(i_t_fft)
q_t = np.fft.ifft(q_w)

paper_norm = normalize(paper_amp, 0, 1)
q_t_norm = normalize(-q_t, 0, 1)

plt.plot(paper_t , paper_norm, label='Paper')
plt.plot(t - 0.1, np.fft.ifftshift(q_t_norm),  label='Q(t)')
plt.xlim(-5, 5)
plt.xlabel('Delay Time (ps)', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.title('Compare',  fontsize=20)
plt.legend()
plt.show()

# plot(np.fft.fftshift(freq), np.fft.fftshift(q_w), 'Frequency', 'Amplitude', 'Q(w)')

