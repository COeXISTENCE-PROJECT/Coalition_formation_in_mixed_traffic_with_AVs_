set N := 0..(Nmax-1); 
set S := 0 .. 2**Nmax - 1;

param a{s in S, i in N} binary;
for {s in S, i in N} {
    let a[s,i] := (s div 2**i) mod 2;
}

set FIXED_ROUTE_0 within N;
set FIXED_ROUTE_1 within N;

param R{i in N,s in S};

var x{s in S} >= 0, <= 1;

maximize total_time:
    sum{s in S, i in N} x[s]*R[i,s];

minimize variance:
    sum{s in S} x[s]*(sum{i in N} R[i,s])**2 - (sum{s in S, i in N} x[s]*R[i,s])**2;

subject to normalization:
    sum{s in S} x[s] == 1;

subject to no_deviation_zero{i in N}:
    sum{s in S : a[s,i] = 0} x[s]*R[i,s] >= sum{s in S : a[s,i] = 0} x[s]*R[i,s+2**i];

subject to no_deviation_one{i in N}:
    sum{s in S : a[s,i] = 1} x[s]*R[i,s] >= sum{s in S : a[s,i] = 1} x[s]*R[i,s-2**i];

subject to fixed_to_route0{s in S, i in FIXED_ROUTE_0: a[s,i] = 1}:
    x[s] == 0;

subject to fixed_to_route1{s in S, i in FIXED_ROUTE_1: a[s,i] = 0}:
    x[s] == 0;