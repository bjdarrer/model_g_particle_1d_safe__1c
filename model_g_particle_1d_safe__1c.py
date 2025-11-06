"""
Model G Particle 1D (SAFE/RESUMABLE)
- Written by Brendan Darrer aided by ChatGPT5 date: 6th November 2025
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb
- with ChatGPT5 writing the code and Brendan guiding to produce a clean code.
- Stiff-stable (BDF) solver
- Progress printing
- MP4 animation (frames rendered incrementally)
- Auto-resume from checkpoints

Tested for: Ubuntu 24 on i7-4790 (Optiplex 7020/9020), Python 3.10+

Install deps:
    pip install numpy scipy matplotlib imageio imageio[ffmpeg]

Run:
    python model_g_particle_1d_safe.py

Notes:
- Uses eqs13 with eqs17 params from the Mathematica notebook.
- Dirichlet at x=+-L/2, zeros initial condition. Forcing chi(x,t) centered at 0.
- Integrates in time segments so we can checkpoint and resume cleanly.
"""
import os
import json
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import imageio.v2 as imageio

# ---------------- Configuration ----------------
# Domain/time
L = 30.0              # total spatial length
Tfinal = 15.0          # total simulated time (use 15 for quick, 30 for full)
segment_dt = 1.0       # integrate in segments of 1.0 time units (checkpoint after each)

# Discretization / output
nx = 401               # spatial points (odd preferred)
nt_anim = 120          # number of animation frames over [0, Tfinal]
max_step = 0.05        # max step for solver (BDF is implicit; this is a soft cap)

# Files / folders
run_name = "model_g_1d_safe__1c"
out_dir = f"out_{run_name}"
frames_dir = os.path.join(out_dir, "frames")
ckpt_path = os.path.join(out_dir, "checkpoint.npz")
mp4_path = os.path.join(out_dir, f"{run_name}.mp4")
final_png = os.path.join(out_dir, "final_snapshot.png")

os.makedirs(frames_dir, exist_ok=True)

# ---------------- Parameters (eqs17) ----------------
params = {
    'a': 14.0,
    'b': 29.0,
    'dx': 1.0,
    'dy': 12.0,
    'p': 1.0,
    'q': 1.0,
    'g': 0.1,
    's': 0.0,
    'u': 0.0,
    'v': 0.0,
    'w': 0.0,
}

# ---------------- Seed forcing chi(x,t) ----------------
def bell(s, x):
    return np.exp(- (x/s)**2 / 2.0)

nseeds = 1
Tseed = 10.0

def chi(x, t):
    if nseeds == 1:
        return -bell(1.0, x) * bell(3.0, t - Tseed)
    elif nseeds == 2:
        return -(bell(1.0, x + 3.303/2) + bell(1.0, x - 3.303/2)) * bell(3.0, t - Tseed)
    else:
        return -(bell(1.0, x + 3.314) + bell(1.0, x) + bell(1.0, x - 3.314)) * bell(3.0, t - Tseed)

# ---------------- Grid & helpers ----------------
xgrid = np.linspace(-L/2, L/2, nx)
dx_space = xgrid[1] - xgrid[0]

# Homogeneous steady-state (G0, X0, Y0)
a = params['a']; b = params['b']; p_par = params['p']; q_par = params['q']; g_par = params['g']; s_par = params['s']; u_par = params['u']; w_par = params['w']
G0 = (a + g_par*w_par) / (q_par - g_par*p_par)
X0 = (p_par*a + q_par*w_par) / (q_par - g_par*p_par)
Y0 = ((s_par*X0**2 + b) * X0 / (X0**2 + u_par)) if (X0**2 + u_par) != 0 else 0.0
print(f"Homogeneous state: G0={G0:.6g}, X0={X0:.6g}, Y0={Y0:.6g}")

# Laplacian and gradient (Dirichlet boundaries)
def laplacian_1d(u):
    dudxx = np.zeros_like(u)
    dudxx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx_space**2)
    return dudxx

def grad_1d(u):
    dudx = np.zeros_like(u)
    dudx[1:-1] = (u[2:] - u[:-2]) / (2*dx_space)
    return dudx

# Pack/unpack helpers
def pack(pG, pX, pY):
    return np.concatenate([pG, pX, pY])

def unpack(y):
    return y[:nx], y[nx:2*nx], y[2*nx:3*nx]

# RHS for solve_ivp
def rhs(t, y_flat):
    pG, pX, pY = unpack(y_flat)
    lapG = laplacian_1d(pG)
    lapX = laplacian_1d(pX)
    lapY = laplacian_1d(pY)
    dGdx = grad_1d(pG)
    dXdx = grad_1d(pX)
    dYdx = grad_1d(pY)

    chi_vec = chi(xgrid, t)

    Xtot = pX + X0
    Ytot = pY + Y0
    nonlinear_s = s_par * (Xtot**3 - X0**3)
    nonlinear_xy = (Xtot**2 * Ytot - X0**2 * Y0)

    dpGdt = lapG - q_par * pG + g_par * pX - params['v'] * dGdx
    dpXdt = params['dx'] * lapX - params['v'] * dXdx + p_par * pG - (1.0 + b) * pX + u_par * pY - nonlinear_s + nonlinear_xy + chi_vec
    dpYdt = params['dy'] * lapY - params['v'] * dYdx + b * pX - u_par * pY + (-nonlinear_xy + nonlinear_s)

    # Dirichlet boundaries: keep edges at zero
    dpGdt[0] = dpGdt[-1] = 0.0
    dpXdt[0] = dpXdt[-1] = 0.0
    dpYdt[0] = dpYdt[-1] = 0.0

    return pack(dpGdt, dpXdt, dpYdt)

# ---------------- Animation/plot helpers ----------------
def plot_snapshot(Yvec, t, savepath=None):
    pG, pX, pY = unpack(Yvec)
    plt.figure(figsize=(10,5))
    plt.plot(xgrid, pY, label='pY (Y)', linewidth=1.5)
    plt.plot(xgrid, pG, label='pG (G)')
    plt.plot(xgrid, pX/10.0, label='pX/10 (X scaled)')
    plt.title(f'Model G 1D — t={t:.3f}')
    plt.xlabel('Space x'); plt.grid(True); plt.legend()
    if savepath:
        plt.savefig(savepath, dpi=120)
        plt.close()
    else:
        plt.show()

# ---------------- Checkpoint logic ----------------
def save_ckpt(t_curr, y_curr, next_frame_idx, frames_done):
    # Ensure both parent folders exist
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    tmp = ckpt_path + ".tmp"

    # Write atomically with verification
    try:
        np.savez_compressed(
            tmp,
            t_curr=t_curr,
            y_curr=y_curr,
            next_frame_idx=next_frame_idx,
            frames_done=np.array(sorted(list(frames_done)), dtype=np.int32)
        )
    except Exception as e:
        print(f"[ERROR] Could not write checkpoint tmp file: {e}")
        return

    # Double-check tmp exists before renaming
    if os.path.exists(tmp):
        os.replace(tmp, ckpt_path)
    else:
        print(f"[WARN] Temporary checkpoint {tmp} missing, skipping rename.")
"""        
def save_ckpt(t_curr, y_curr, next_frame_idx, frames_done):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)  # ensure directory exists
    tmp = ckpt_path + ".tmp"
    np.savez_compressed(tmp,
        t_curr=t_curr,
        y_curr=y_curr,
        next_frame_idx=next_frame_idx,
        frames_done=np.array(sorted(list(frames_done)), dtype=np.int32)
    )
    os.replace(tmp, ckpt_path)
    
    
def save_ckpt(t_curr, y_curr, next_frame_idx, frames_done):
    os.makedirs(out_dir, exist_ok=True)
    tmp = ckpt_path + ".tmp"
    #np.savez_compressed(tmp, t_curr=t_curr, y_curr=y_curr, next_frame_idx=next_frame_idx,
    #                    frames_done=np.array(frames_done, dtype=np.int32)) # BJD commented out 6.11.2025 18:12   
    np.savez_compressed(tmp,
    t_curr=t_curr,
    y_curr=y_curr,
    next_frame_idx=next_frame_idx,
    frames_done=np.array(sorted(list(frames_done)), dtype=np.int32)
    )
    os.replace(tmp, ckpt_path)
"""

def load_ckpt():
    if not os.path.exists(ckpt_path):
        return None
    data = np.load(ckpt_path, allow_pickle=True)
    return {
        't_curr': float(data['t_curr']),
        'y_curr': data['y_curr'],
        'next_frame_idx': int(data['next_frame_idx']),
        'frames_done': set(map(int, data['frames_done'].tolist() if hasattr(data['frames_done'], 'tolist') else list(data['frames_done'])))
    }

# ---------------- Main integration loop (segmented, resumable) ----------------

def main():
    # Prepare desired frame times
    frame_times = np.linspace(0.0, Tfinal, nt_anim)

    # Resume or start fresh
    ck = load_ckpt()
    if ck is None:
        t_curr = 0.0
        y_curr = np.zeros(3*nx)
        next_frame_idx = 0
        frames_done = set()
        print("[Start] Fresh run")
    else:
        t_curr = ck['t_curr']
        y_curr = ck['y_curr']
        next_frame_idx = ck['next_frame_idx']
        frames_done = ck['frames_done']
        print(f"[Resume] t={t_curr:.3f}, next_frame={next_frame_idx}/{nt_anim}, frames_done={len(frames_done)}")

    t_start_wall = time.time()

    # Render any frames at t=0 if needed
    while next_frame_idx < nt_anim and frame_times[next_frame_idx] <= t_curr + 1e-12:
        tframe = frame_times[next_frame_idx]
        fpath = os.path.join(frames_dir, f"frame_{next_frame_idx:04d}.png")
        if next_frame_idx not in frames_done:
            plot_snapshot(y_curr, tframe, savepath=fpath)
            frames_done.add(next_frame_idx)
        next_frame_idx += 1
        save_ckpt(t_curr, y_curr, next_frame_idx, frames_done)

    # Time stepping by segments
    while t_curr < Tfinal - 1e-12:
        t_seg_end = min(Tfinal, t_curr + segment_dt)
        print(f"[Integrate] {t_curr:.3f} -> {t_seg_end:.3f} (segment_dt={segment_dt})")
        seg_sol = solve_ivp(rhs, (t_curr, t_seg_end), y_curr, method='BDF',
                            max_step=max_step, atol=1e-6, rtol=1e-6, dense_output=True)
        if seg_sol.status < 0:
            print("[WARN] Segment integration reported failure:", seg_sol.message)
        y_curr = seg_sol.y[:, -1]
        t_curr = seg_sol.t[-1]

        # Render any frames whose times fall within this segment
        # Use dense output to evaluate at precise times
        if seg_sol.sol is not None:
            while next_frame_idx < nt_anim and frame_times[next_frame_idx] <= t_curr + 1e-12:
                tframe = frame_times[next_frame_idx]
                y_at = seg_sol.sol(tframe)
                fpath = os.path.join(frames_dir, f"frame_{next_frame_idx:04d}.png")
                if next_frame_idx not in frames_done:
                    plot_snapshot(y_at, tframe, savepath=fpath)
                    frames_done.add(next_frame_idx)
                next_frame_idx += 1
                save_ckpt(t_curr, y_curr, next_frame_idx, frames_done)
        else:
            # Fallback: render with current state if dense output unavailable
            while next_frame_idx < nt_anim and frame_times[next_frame_idx] <= t_curr + 1e-12:
                tframe = frame_times[next_frame_idx]
                fpath = os.path.join(frames_dir, f"frame_{next_frame_idx:04d}.png")
                if next_frame_idx not in frames_done:
                    plot_snapshot(y_curr, tframe, savepath=fpath)
                    frames_done.add(next_frame_idx)
                next_frame_idx += 1
                save_ckpt(t_curr, y_curr, next_frame_idx, frames_done)

        # Periodic progress print
        elapsed = time.time() - t_start_wall
        print(f"  -> Reached t={t_curr:.3f} / {Tfinal}, frames={len(frames_done)}/{nt_anim}, wall={elapsed:.1f}s")

        # Save last snapshot as a convenience
        plot_snapshot(y_curr, t_curr, savepath=final_png)
        save_ckpt(t_curr, y_curr, next_frame_idx, frames_done)

    # Finalize MP4
    print("[Video] Writing MP4:", mp4_path)
    with imageio.get_writer(mp4_path, fps=max(8, int(nt_anim / max(1, Tfinal/2)))) as writer:
        for i in range(nt_anim):
            f = os.path.join(frames_dir, f"frame_{i:04d}.png")
            img = imageio.imread(f)
            writer.append_data(img)
    print("[Done] MP4 saved.")
    print("Final snapshot:", final_png)

if __name__ == '__main__':
    main()
