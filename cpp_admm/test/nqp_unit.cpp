
#include "../src/csv.hpp"
#include "../src/model.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cstdio>
#include <vector>
int main(int argc, char **argv) {
    std::string inF = argv[1], conF = argv[2], wtF = argv[3];
    std::string err;
    auto inputs = loadInputs(inF, err);
    auto constraints = loadConstraints(conF, err);
    auto weights = loadWeights(wtF, err);
    step1Relax(inputs);
    InputsCols cols = toColumns(inputs);
    auto det = assembleConstraints(cols.univ, constraints, weights);
    Problem p = buildProblem(cols, det, false, 100.0);
    scaleRows(p);
    int n = p.n, m = (int)p.rows.size();
    int gs = p.generalStart;
    double gamma = 1e7, sigma0 = 1.0;
    std::vector<char> active(m,0);
    for (int r = 0; r < m; r++)
        active[r] = (0.0 <= p.rows[r].lb) || (0.0 >= p.rows[r].ub);
    int nAct = (int)std::count(active.begin(),active.end(),1);
    printf("active rows at start: %d / %d\n", nAct, m);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);
    for (int j = 0; j < n; j++) M(j,j) = p.pDiag[j] + 1.0/gamma;
    for (int r = 0; r < m; r++) {
        if (!active[r]) continue;
        const Row &rw = p.rows[r];
        for (size_t a = 0; a < rw.idx.size(); a++)
            for (size_t b = 0; b < rw.idx.size(); b++)
                M(rw.idx[a], rw.idx[b]) += sigma0 * rw.val[a]*rw.val[b];
    }
    printf("M diag min %.3e max %.3e\n", M.diagonal().minCoeff(), M.diagonal().maxCoeff());
    Eigen::VectorXd b(n);
    for (int j = 0; j < n; j++) b[j] = -p.q[j];
    for (int r = 0; r < m; r++) {
        double z0 = 0 < p.rows[r].lb ? p.rows[r].lb : (0 > p.rows[r].ub ? p.rows[r].ub : 0);
        double yh = sigma0 * (0.0 - z0);
        if (yh == 0) continue;
        for (size_t t = 0; t < p.rows[r].idx.size(); t++)
            b[p.rows[r].idx[t]] -= p.rows[r].val[t] * yh;
    }
    Eigen::VectorXd dRef = M.partialPivLu().solve(b);
    printf("dense LU Newton dir: |d|max = %.4e\n", dRef.cwiseAbs().maxCoeff());
    // union-find over active structure pair rows
    std::vector<int> uf(n); for (int j=0;j<n;j++) uf[j]=j;
    auto find=[&](int xx){while(uf[xx]!=xx){uf[xx]=uf[uf[xx]];xx=uf[xx];}return xx;};
    for (int r = 0; r < gs; r++) {
        if (!active[r] || p.rows[r].idx.size()!=2) continue;
        int a=find(p.rows[r].idx[0]), bb=find(p.rows[r].idx[1]);
        if (a!=bb) uf[a]=bb;
    }
    std::vector<int> rootBlk(n,-1); std::vector<std::vector<int>> blkVars;
    for (int j=0;j<n;j++){int rt=find(j); if(rootBlk[rt]<0){rootBlk[rt]=blkVars.size();blkVars.push_back({});} blkVars[rootBlk[rt]].push_back(j);}
    int nBlk=blkVars.size();
    printf("blocks: %d (sizes: 1x1=%d, 2x2=%d, 3x3=%d)\n", nBlk,
        (int)std::count_if(blkVars.begin(),blkVars.end(),[](auto&v){return v.size()==1;}),
        (int)std::count_if(blkVars.begin(),blkVars.end(),[](auto&v){return v.size()==2;}),
        (int)std::count_if(blkVars.begin(),blkVars.end(),[](auto&v){return v.size()==3;}));
    std::vector<double> blkInv(nBlk*16,0.0);
    for (int bi=0;bi<nBlk;bi++){
        int sz=blkVars[bi].size();
        int root=find(blkVars[bi][0]);
        Eigen::MatrixXd B=Eigen::MatrixXd::Zero(sz,sz);
        for (int u=0;u<sz;u++){int var=blkVars[bi][u];B(u,u)=p.pDiag[var]+1.0/gamma;}
        for (int r=0;r<gs && false;r++){} // rows handled below generically
        for (int r=0;r<gs;r++){
            if(!active[r]||p.rows[r].idx.empty())continue;
            if(find(p.rows[r].idx[0])!=root)continue;
            const Row&rw=p.rows[r];
            for(size_t a=0;a<rw.idx.size();a++)for(size_t c=0;c<rw.idx.size();c++){
                int va=rw.idx[a],vc=rw.idx[c];int ua=-1,uc=-1;
                for(int u=0;u<sz;u++){if(blkVars[bi][u]==va)ua=u;if(blkVars[bi][u]==vc)uc=u;}
                if(ua>=0&&uc>=0)B(ua,uc)+=sigma0*rw.val[a]*rw.val[c];
            }
        }
        Eigen::MatrixXd Bi=B.inverse();
        for(int u=0;u<sz;u++)for(int v=0;v<sz;v++)blkInv[bi*16+u*4+v]=Bi(u,v);
    }
    std::vector<int> wRow; for(int i=gs;i<m;i++) if(active[i]) wRow.push_back(i);
    int k=wRow.size(); printf("active general rows: %d\n",k);
    Eigen::MatrixXd W=Eigen::MatrixXd::Zero(k,n);
    for(int i=0;i<k;i++){const Row&rw=p.rows[wRow[i]];for(size_t t=0;t<rw.idx.size();t++)W(i,rw.idx[t])=std::sqrt(sigma0)*rw.val[t];}
    Eigen::VectorXd v1(n);
    for(int j=0;j<n;j++)v1[j]=b[j];
    for(int bi=0;bi<nBlk;bi++){int sz=blkVars[bi].size();
        Eigen::VectorXd bv(sz),rr(sz);
        for(int u=0;u<sz;u++)bv[u]=v1[blkVars[bi][u]];
        for(int u=0;u<sz;u++){double s2=0;for(int v=0;v<sz;v++)s2+=blkInv[bi*16+u*4+v]*bv[v];rr[u]=s2;}
        for(int u=0;u<sz;u++)v1[blkVars[bi][u]]=rr[u];}
    Eigen::VectorXd dW(n);
    if(k==0){dW=v1;}
    else{
        Eigen::MatrixXd Dinv=Eigen::MatrixXd::Zero(n,n);
        for(int bi=0;bi<nBlk;bi++){int sz=blkVars[bi].size();
            for(int u=0;u<sz;u++)for(int v=0;v<sz;v++)Dinv(blkVars[bi][u],blkVars[bi][v])=blkInv[bi*16+u*4+v];}
        Eigen::MatrixXd S=Eigen::MatrixXd::Identity(k,k)+W*Dinv*W.transpose();
        Eigen::VectorXd rhs2=W*v1;
        Eigen::VectorXd v=S.llt().solve(rhs2);
        dW=v1-Dinv*W.transpose()*v;
    }
    printf("block+Woodbury vs dense LU: max|diff| = %.3e\n", (dW-dRef).cwiseAbs().maxCoeff());
    return 0;
}
