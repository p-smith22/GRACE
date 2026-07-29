# Import package:
import numpy as np

# Define potential flow path:
def potential_path(start, goal, OBS, R, step=0.03, maxsteps=6000):

    p=np.array(start,float)
    path=[p.copy()]
    for _ in range(maxsteps):
        to_goal=goal-p;dg=np.linalg.norm(to_goal)
        if dg<step*2: break
        v=to_goal/dg
        for o in OBS:
            d=p-o;dist=np.linalg.norm(d);infl=R*3.5
            if dist<infl:
                n=d/max(dist,1e-6)
                v=v+n*(R/max(dist,0.25))**2*1.2
                perp=np.array([-n[1],n[0]])
                if perp@(goal-p)<0: perp=-perp
                v=v+perp*(R/max(dist,0.25))**2*1.6
        vn=np.linalg.norm(v)
        if vn<1e-6: break
        p=p+step*v/vn;path.append(p.copy())
    path.append(np.array(goal,float))
    return np.array(path)

def track(s,tgt,OBS,R_route,pi,tidx,ref,u_lo,u_hi,iters=60):
    # heavy path-tracking + endpoint; NO min-effort relax (feasibility first)
    N,nu,nx=s.N,s.nu,s.nx;zt=s.target(tgt);z0=np.asarray(s.z0,float)
    lo=None if u_lo is None else np.tile(u_lo,N);hi=None if u_hi is None else np.tile(u_hi,N)
    clip=lambda U: U if lo is None else np.clip(U,lo,hi)
    U=np.zeros(N*nu);Rr=np.eye(nu)*1e-3  # tiny effort weight (feasibility priority)
    def roll(U): return s.rollout(U)
    def enderr(U): z=roll(U)[-1];return np.array([z[t]-zt[j] for j,t in enumerate(tidx)])
    def pathdev(U): p=roll(U)[:,pi];return np.sum((p-ref[:len(p)])**2)
    wend=5e4;wtrack=3.0
    for _ in range(iters):
        Z=roll(U);p=Z[:,pi]
        # One batched call for the whole tape of one-step Jacobians (falls back to the
        # per-node call if the system was built without the batched version):
        if getattr(s,'step_jac_all',None) is not None:
            A,B=s.step_jac_all(Z,U)
        else:
            A=[np.asarray(s.step_jac(Z[k],U[k*nu:(k+1)*nu])[0]) for k in range(N)]
            B=[np.asarray(s.step_jac(Z[k],U[k*nu:(k+1)*nu])[1]) for k in range(N)]
        Qk=[np.zeros((nx,nx)) for _ in range(N+1)];qk=[np.zeros(nx) for _ in range(N+1)]
        for k in range(N+1):
            for a2 in range(2):
                Qk[k][pi[a2],pi[a2]]+=2*wtrack;qk[k][pi[a2]]+=2*wtrack*(Z[k,pi[a2]]-ref[k,a2])
        P=np.zeros((nx,nx));sv=np.zeros(nx)
        for j,t in enumerate(tidx): P[t,t]=wend;sv[t]=wend*(Z[-1,t]-zt[j])
        Ks=[None]*N;ks=[None]*N
        for k in range(N-1,-1,-1):
            Ak,Bk=A[k],B[k];Quu=Rr+Bk.T@P@Bk+1e-2*np.eye(nu);Qux=Bk.T@P@Ak;qu=Bk.T@sv+Rr@U[k*nu:(k+1)*nu]
            Kk=np.linalg.solve(Quu,Qux);kk=np.linalg.solve(Quu,qu);Ks[k]=Kk;ks[k]=kk
            P=Qk[k]+Ak.T@P@Ak-Qux.T@Kk;sv=qk[k]+Ak.T@sv-Qux.T@kk
        def merit(U): ee=enderr(U);return wtrack*pathdev(U)+wend*float(ee@ee)+float(U@U)*1e-3
        m0=merit(U);best=None
        for a in [1.,0.5,0.25,0.1,0.05,0.02,0.01]:
            Un=U.copy();zn=z0.copy();ok=True
            for k in range(N):
                du=-a*ks[k]-Ks[k]@(zn-Z[k]);Un[k*nu:(k+1)*nu]=U[k*nu:(k+1)*nu]+du;Un=clip(Un);zn=s.step_np(zn,Un[k*nu:(k+1)*nu])
                if not np.all(np.isfinite(zn)): ok=False;break
            if ok and merit(Un)<m0-1e-9: best=Un;break
        if best is None: break
        U=best
    return U

def feasible_solve(s,tgt,OBS,R,pi,tidx,u_lo=None,u_hi=None,verbose=False):
    # route around R_route, track, inflate R_route until tracked traj clears TRUE R with no penetration
    zt=s.target(tgt);g=zt[[list(tidx).index(pi[0]),list(tidx).index(pi[1])]];N=s.N
    def clr(U): p=s.rollout(U)[:,pi];return min((np.sum((p-o)**2,axis=1)**.5).min() for o in OBS)
    def ee(U): z=s.rollout(U)[-1];return np.linalg.norm([z[t]-zt[j] for j,t in enumerate(tidx)])
    R_route=R
    for attempt in range(8):
        path=potential_path(s.z0[pi],g,OBS,R_route)
        pl=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]);pl/=max(pl[-1],1e-9)
        ref=np.array([path[min(np.searchsorted(pl,k/N),len(path)-1)] for k in range(N+1)])
        U=track(s,tgt,OBS,R_route,pi,tidx,ref,u_lo,u_hi)
        c=clr(U);e=ee(U)
        if verbose: print('  attempt%d R_route%.2f -> clr%.3f penet%.3f endpt%.3f'%(attempt,R_route,c,max(0,R-c),e))
        if c>=R and e<0.05: return U,R_route,c,e
        R_route+=max(R-c,0)+0.15  # inflate by penetration depth
    return U,R_route,clr(U),ee(U)

def feasible_solve2(s,tgt,OBS,R,pi,tidx,u_lo=None,u_hi=None):
    # Phase 1: feasible position track (from feasible_solve). Phase 2: hold clearance, drive FULL endpoint.
    U,R_route,cl,ee=feasible_solve(s,tgt,OBS,R,pi,tidx,u_lo,u_hi)
    N,nu,nx=s.N,s.nu,s.nx;zt=s.target(tgt);z0=np.asarray(s.z0,float)
    lo=None if u_lo is None else np.tile(u_lo,N);hi=None if u_hi is None else np.tile(u_hi,N)
    clip=lambda U: U if lo is None else np.clip(U,lo,hi)
    def roll(U): return s.rollout(U)
    def clr(U): p=roll(U)[:,pi];return min((np.sum((p-o)**2,axis=1)**.5).min() for o in OBS)
    def obsviol(U,m):
        p=roll(U)[:,pi];v=0.
        for o in OBS: v+=np.sum(np.maximum(0,(R+m)-np.sqrt(np.sum((p-o)**2,axis=1)))**2)
        return v
    def enderr(U): z=roll(U)[-1];return np.array([z[t]-zt[j] for j,t in enumerate(tidx)])
    Rr=np.eye(nu)*1e-3
    # phase 2: obstacle penalty (hold clearance) + strong full endpoint, margin keeps it from re-penetrating
    margin=max(0.15,(cl-R)*0.5)  # keep the margin the feasible track achieved
    for it in range(50):
        Z=roll(U);p=Z[:,pi]
        # One batched call for the whole tape of one-step Jacobians (falls back to the
        # per-node call if the system was built without the batched version):
        if getattr(s,'step_jac_all',None) is not None:
            A,B=s.step_jac_all(Z,U)
        else:
            A=[np.asarray(s.step_jac(Z[k],U[k*nu:(k+1)*nu])[0]) for k in range(N)]
            B=[np.asarray(s.step_jac(Z[k],U[k*nu:(k+1)*nu])[1]) for k in range(N)]
        Qk=[np.zeros((nx,nx)) for _ in range(N+1)];qk=[np.zeros(nx) for _ in range(N+1)]
        for k in range(N+1):
            for o in OBS:
                d=p[k]-o;dist=np.sqrt(np.sum(d**2)+1e-9);viol=(R+margin)-dist
                if viol>0:
                    n=d/dist;rho=120.
                    qk[k][pi[0]]+=-2*rho*viol*n[0];qk[k][pi[1]]+=-2*rho*viol*n[1];Qxy=2*rho*np.outer(n,n)
                    for a2 in range(2):
                        for b2 in range(2): Qk[k][pi[a2],pi[b2]]+=Qxy[a2,b2]
        P=np.zeros((nx,nx));sv=np.zeros(nx);wend=1e5
        for j,t in enumerate(tidx): P[t,t]=wend;sv[t]=wend*(Z[-1,t]-zt[j])
        Ks=[None]*N;ks=[None]*N
        for k in range(N-1,-1,-1):
            Ak,Bk=A[k],B[k];Quu=Rr+Bk.T@P@Bk+1e-2*np.eye(nu);Qux=Bk.T@P@Ak;qu=Bk.T@sv+Rr@U[k*nu:(k+1)*nu]
            Kk=np.linalg.solve(Quu,Qux);kk=np.linalg.solve(Quu,qu);Ks[k]=Kk;ks[k]=kk
            P=Qk[k]+Ak.T@P@Ak-Qux.T@Kk;sv=qk[k]+Ak.T@sv-Qux.T@kk
        def merit(U): ee=enderr(U);return float(U@U)*1e-3+1e4*obsviol(U,0.0)+wend*float(ee@ee)
        m0=merit(U);best=None
        for a in [1.,0.5,0.25,0.1,0.05,0.02,0.01]:
            Un=U.copy();zn=z0.copy();ok=True
            for k in range(N):
                du=-a*ks[k]-Ks[k]@(zn-Z[k]);Un[k*nu:(k+1)*nu]=U[k*nu:(k+1)*nu]+du;Un=clip(Un);zn=s.step_np(zn,Un[k*nu:(k+1)*nu])
                if not np.all(np.isfinite(zn)): ok=False;break
            # only accept if STILL clean (no penetration)
            if ok and clr(Un)>=R and merit(Un)<m0-1e-9: best=Un;break
        if best is None: break
        U=best
    return U

def polish(s,tgt,OBS,R,pi,tidx,U,u_lo=None,u_hi=None):
    # stationarity polish: reduce effort in the endpoint null space (lambda-style),
    # barrier vetoes penetration, endpoint held by reprojection. Starts from a feasible U.
    N,nu,nx=s.N,s.nu,s.nx;zt=s.target(tgt);m=len(tidx)
    lo=None if u_lo is None else np.tile(u_lo,N);hi=None if u_hi is None else np.tile(u_hi,N)
    clip=lambda U: U if lo is None else np.clip(U,lo,hi)
    def roll(U): return s.rollout(U)
    def clr(U): p=roll(U)[:,pi];return min((np.sum((p-o)**2,axis=1)**.5).min() for o in OBS)
    def enderr(U): return s.endpoint(U)-zt
    # barrier-safe endpoint reprojection
    def reproj(U,it=12):
        for _ in range(it):
            r=enderr(U)
            if np.linalg.norm(r)<1e-7: break
            _,J=s.endpoint_jac(U);full=-J.T@np.linalg.solve(J@J.T+1e-10*np.eye(m),r);acc=False
            for a in [1.,0.5,0.25,0.1,0.05]:
                Ut=clip(U+a*full)
                if clr(Ut)>=R-1e-9: U=Ut;acc=True;break
            if not acc: break
        return U
    c=float(U@U);tr=0.3;stall=0
    for it in range(400):
        ev,Co=s.endpoint_jac(U)
        # Project the effort gradient into the null space of ALL active constraints: the
        # endpoint rows AND the obstacle rows that are currently on the boundary. Including
        # the active obstacle rows lets the descent SLIDE ALONG the boundary instead of
        # being rejected whenever the cheapest direction presses into the obstacle.
        Z=roll(U);p=Z[:,pi];Jp=np.array(s.pos_jac(U))
        rows=[Co]
        for o in OBS:
            dist=np.sqrt(np.sum((p-o)**2,axis=1))
            # Only nodes essentially ON the boundary are treated as active equality rows.
            # Nodes with real slack are left free so the descent can still settle them
            # inward onto the true radius rather than freezing an unnecessary standoff:
            for k in np.where(dist<R+1e-3)[0]:
                nrm=(p[k]-o)/max(dist[k],1e-9)
                rows.append((nrm@Jp[k]).reshape(1,-1))
        A=np.vstack(rows);WA=A@A.T+1e-10*np.eye(A.shape[0])
        gc=2*U  # pure effort gradient (no barrier term -> descend toward true min-effort)
        Rg=gc-A.T@np.linalg.solve(WA,A@gc)
        if np.linalg.norm(Rg)/max(np.linalg.norm(gc),1e-9)<1e-4: break
        d=-Rg/np.linalg.norm(Rg)
        Ut=reproj(clip(U+tr*np.linalg.norm(U)*d))
        # accept only if still clean, endpoint held, and cost decreased
        if clr(Ut)>=R-1e-9 and np.linalg.norm(enderr(Ut))<1e-4 and float(Ut@Ut)<c-1e-12:
            U=Ut;c=float(U@U);tr=min(tr*1.5,1.0);stall=0
        else:
            tr*=0.5;stall+=1
            # A collapsed trust region usually means the descent direction was blocked by the
            # constraint geometry rather than that we are optimal -- restart the region a few
            # times before giving up, which escapes stalls without loosening any tolerance:
            if tr<1e-7:
                if stall>5: break
                tr=0.3
    return U



# Entry point with the GRACE interface: minimum-effort shoot avoiding obstacles.
def lambda_obstacle(system, z_target, obstacles, R, pos_idx=(0, 1), max_it=250, reg=1e-8,
                    u_lo=None, u_hi=None, R_weights=None):

    # Gather geometry and run the feasible-then-optimal pipeline:
    pi = list(pos_idx)

    # The position indices must be part of the endpoint target so the router can read the
    # goal position; give a clear message rather than an index error if they are not:
    missing = [i for i in pi if i not in list(system.tidx)]
    if missing:
        raise ValueError(
            "obstacle avoidance needs pos_idx %s to be included in the system's target_idx %s "
            "(missing %s) -- rebuild the system with those position states in target_idx, "
            "or pass the correct pos_idx." % (tuple(pi), tuple(system.tidx), tuple(missing)))
    OBS = [np.asarray(o, float) for o in obstacles]
    tidx = list(system.tidx)

    # Feasible (zero-penetration) trajectory, then stationarity polish to the optimum:
    U0 = feasible_solve2(system, z_target, OBS, R, pi, tidx, u_lo, u_hi)
    U = polish(system, z_target, OBS, R, pi, tidx, U0, u_lo, u_hi)

    # Feasibility check: flag if the result still penetrates or misses the endpoint, which
    # means the request is too hard for this horizon (needs more time or an easier obstacle):
    Z = system.rollout(U)
    clr = min((np.sum((Z[:, pi] - o) ** 2, axis=1) ** 0.5).min() for o in OBS)
    zt = system.target(z_target)
    ee = np.linalg.norm(np.array([Z[-1, t] - zt[j] for j, t in enumerate(tidx)]))
    system._obstacle_infeasible = bool(clr < R - 1e-2 or ee > 5e-2)
    if system._obstacle_infeasible:
        print("[grace] warning: obstacle request appears infeasible for this horizon "
              "(clearance %.2f vs R %.2f, endpoint error %.2e). Try a longer horizon N, "
              "more time, or an easier obstacle." % (clr, R, ee))
        print("[GRACE] WARNING: obstacle request appears infeasible for this horizon. "
              "Try a longer horizon N, more time, or an easier obstacle.")
    return U
