cc ******************************************************************************************
cc
cc    ****************************************************
cc    *      COPYRIGHT: YI, ZHANG                        *
cc    *      LASSP, CORNELL UNIVERSITY                   *
cc    *      All Rights Reserved.    ----   2016. 09     *
cc    ****************************************************
cc
cc ******************************************************************************************

	implicit none
cc	Parameter of the neural network. nn: number of neurons in the fully connected hidden layer;
cc	ni: number of inputs; L: LxL is the 2-d input size; mini: mini-patch size; 	
cc	epoch: size of the epoch; itval: interval between two entries to ensure no correlation. 
cc	ncount: the times eta is halved; nstop, the limit of ncount before optimization terminates. 
	integer nn, ni, L, mini, epoch, itval, nepoch, ncount, nstop
	parameter (nn=10, L=30, ni=L*L, mini=10, epoch=2000, itval=L*L/2)
cc	The weights and biases. 
	double precision w1(ni, nn), b1(nn), w2(nn), b2
	double precision dw1(ni, nn), db1(nn), dw2(nn), db2
cc	The inputs: low and high temperature phases. vx1 and vx2 are validation data groups. 
	integer x1(L, L), x2(L, L), nval, nsap, accu, nstep
	parameter (nval=4000, nsap=20000, nstop=2, nstep=5) 
	integer vx1(L, L, nsap+nval), vx2(L, L, nsap+nval)
	double precision a1(nn), a2, delta1(nn), delta2, z1(nn), z2
cc	Hyper-parameters. eta: learning speed; lambda: L2 regulation; accu: accuracy. 
	double precision eta, lambda, cost
cc	Parameter for the input-output models. T1: low temperature; T2: high temperature; T: test temperature. 
	double precision T1, T2, T, engy
	parameter (T1 = 1.0, T2 = 4.5)
cc	Running parameters. 
	integer i, j, ip1, im1, jp1, jm1, info, k, m, n
	double precision d1, d2, d, rand
cc	Randam number parameters. 
	integer*4 timearray(3)
	eta = 0.16
	lambda = 0.00001
cc	Initialize the random numbers.
	call itime(timearray)
	call SRAND(timearray(1)*3600+
     &	timearray(2)*60+timearray(3))

cc	Initializing the neural network with Box-Muller transformation for Gaussian distribution. 
	do 100 i = 1, nn
cc	Bias (weight) initialization with standard deviation of 1 (1/sqrt(input)). 
110	d1 = 2.0 * rand() - 1.0
	d2 = 2.0 * rand() - 1.0
	d = d1*d1+d2*d2
	if(d .ge. 1.0) goto 110
	d = sqrt((-2.0 * log(d))/d)
	b1(i) = d1*d
	w2(i) = d2*d/sqrt(1.0*nn)
	do 120 j = 1, ni/2
130	d1 = 2.0 * rand() - 1.0
	d2 = 2.0 * rand() - 1.0
	d = d1*d1+d2*d2
	if(d .ge. 1.0) goto 130
	d = sqrt((-2.0 * log(d))/d)
	w1(j*2-1, i) = d1*d/sqrt(1.0*ni)
	w1(j    , i) = d2*d/sqrt(1.0*ni)
120	continue
100	continue
140	d1 = 2.0 * rand() - 1.0
	d2 = 2.0 * rand() - 1.0
	d = d1*d1+d2*d2
	if(d .ge. 1.0) goto 140
	d = sqrt((-2.0 * log(d))/d)
	b2 = d1*d

cc	Initializing the input. 
	do 150 i = 1, L
	do 150 j = 1, L
	if(rand() .gt. 0.5) then
	x1(i, j) = 1
	x2(i, j) = 1
	else
	x1(i, j) = -1
	x2(i, j) = -1
	endif
150	continue
cc	Monte Carlo sampling of 100 steps to converge into the phases. 
	do 160 m = 1, 2000+nval+nsap
	do 170 k = 1, itval
cc	Random coordinate for spin flip. 
	i = rand() * L+1
	j = rand() * L+1
	if(i .gt. L) i = i - 1
	if(j .gt. L) j = j - 1
cc	Periodic boundary conditions. 
	ip1 = i+1
	if(ip1 .gt. L) ip1 = ip1 - L
	im1 = i-1
	if(im1 .le. 0) im1 = im1 + L
	jp1 = j+1
	if(jp1 .gt. L) jp1 = jp1 - L
	jm1 = j-1
	if(jm1 .le. 0) jm1 = jm1 + L

cc	Energy for square lattice. 
	engy = - x1(i, j) * (x1(ip1,j)+x1(im1,j)+x1(i,jp1)+x1(i,jm1))
	if(engy .gt. 0.0) then
cc	Accept the flip. 
	x1(i, j) = - x1(i, j)
	else
cc	Flip with probability by Boltzmann weight.
	if(rand() .lt. exp(engy*2.0/T1)) x1(i, j) = - x1(i, j)
	endif
	engy = - x2(i, j) * (x2(ip1,j)+x2(im1,j)+x2(i,jp1)+x2(i,jm1))
	if(engy .gt. 0.0) then
cc	Accept the flip. 
	x2(i, j) = - x2(i, j)
	else
	if(rand() .lt. exp(engy*2.0/T2)) x2(i, j) = - x2(i, j)
	endif
170	continue
cc	Produce the sample & validation data groups. 
	if(m .gt. 2000) then
	info = 1
	if(rand() .lt. 0.5) info = -1
	do 180 i = 1, L
	do 180 j = 1, L
	vx1(i, j, m-2000) = x1(i, j)*info
	vx2(i, j, m-2000) = x2(i, j)*info
180	continue
	endif
160	continue
	ncount = 0
	nepoch = 0
cc	Initialization complete. 

cc	Learning loops. 
200	nepoch = nepoch + 1
	cost = 0.0
	do 290 n = 1, epoch
	do 230 i = 1, nn
	db1(i) = 0.0
	dw2(i) = 0.0
	do 230 j = 1, ni
	dw1(j, i) = 0.0
230	continue
	db2 = 0.0
	do 220 m = 1, mini
cc	Ready to update weights and biases. 
	info = nsap* rand()+1
	if(rand() .gt. 0.5) then
	do 210 i = 1, L
	do 210 j = 1, L
	x1(i, j) = vx1(i, j, info)
210	continue
	info = 1.0
	else
	do 215 i = 1, L
	do 215 j = 1, L
	x1(i, j) = vx2(i, j, info)
215	continue
	info = 0.0
	endif
cc	Input generated. Feed it forward. 
	z2 = 0.0
	do 240 k = 1, nn
	z1(k) = 0.0
	do 250 j = 1, L	
	do 250 i = 1, L
cc	Randomize the input configurations. 
	z1(k) = z1(k) + x1(i, j) * w1((j-1)*L+i, k)
250	continue
	z1(k) = z1(k) + b1(k)
	a1(k) = 1.0/(1.0+ exp(-z1(k)))
	z2 = z2 + w2(k) * a1(k) 
240	continue
	z2 = z2 + b2
	a2 = 1.0/(1.0+ exp(-z2))
	cost = cost - info*log(a2) - (1-info)*log(1.0-a2)
cc	Calculate the errors. 
	delta2 = a2 - info*1.0
cc	Back propagation. 
	do 260 i = 1, nn
	delta1(i) = delta2 * w2(i) * a1(i) * (1.0 - a1(i)) 
260	continue
cc	Calculate the gradient function. 	
	do 270 i = 1, nn
	db1(i) = db1(i) + delta1(i) 
	dw2(i) = dw2(i) + delta2 * a1(i)
	do 270 j = 1, L
	do 270 k = 1, L
	dw1((k-1)*L+j, i) = dw1((k-1)*L+j, i) + delta1(i) * x1(j, k)
270	continue
	db2 = db2 + delta2
220	continue
cc	Update the new weights and biases. 
	do 280 i = 1, nn
	b1(i) = b1(i) - db1(i) * eta  / mini
	w2(i) = w2(i)*(1.0-lambda) - dw2(i) * eta / mini
	do 280 j = 1, ni
	w1(j, i) = w1(j, i)*(1.0-lambda) - dw1(j, i) * eta / mini
280	continue
	b2 = b2 - db2 * eta / mini
290	continue
cc	Epoch completed. 

	accu = 0
	do 310 m=nsap+1, nsap+nval
	z2 = 0.0
	do 320 k = 1, nn
	z1(k) = 0.0
	do 330 j = 1, L	
	do 330 i = 1, L
	z1(k) = z1(k) + vx1(i, j, m) * w1((j-1)*L+i, k)
330	continue
	z1(k) = z1(k) + b1(k)
	a1(k) = 1.0/(1.0+ exp(-z1(k)))
	z2 = z2 + w2(k) * a1(k) 
320	continue
	z2 = z2 + b2
	if(z2 .gt. 0) accu = accu + 1
	z2 = 0.0
	do 340 k = 1, nn
	z1(k) = 0.0
	do 350 j = 1, L	
	do 350 i = 1, L
	z1(k) = z1(k) + vx2(i, j, m) * w1((j-1)*L+i, k)
350	continue
	z1(k) = z1(k) + b1(k)
	a1(k) = 1.0/(1.0+ exp(-z1(k)))
	z2 = z2 + w2(k) * a1(k) 
340	continue
	z2 = z2 + b2
	if(z2 .lt. 0) accu = accu + 1
310	continue
	print*, 'Epoch', nepoch, 'accuracy =', accu*0.5/nval
	print*, 'Slowdown #=', ncount, 'cost=', cost/epoch/mini
cc	Self-termination and slow-down routine. 
	if(nepoch/nstep*nstep .eq. nepoch .AND.
     &	 accu*0.5/nval .gt. 0.95) then
	ncount = ncount + 1
	eta = eta / 2
c	lambda = lambda / 2
	endif
	if(ncount .le. nstop) goto 200
	print*, 'Training complete.'

cc	Calculate the phase transition. 
	print*, 'Calculating phase diagram.'

	do 420 i = 1, L
	do 420 j = 1, L
	if(rand() .gt. 0.5) then
	x1(i, j) = 1
	else
	x1(i, j) = -1
	endif
420	continue

	do 410 n = 0, 100
	T = 3.5 - n * 0.025
	accu = 0
	do 430 m = 1, nval
	do 440 k = 1, itval
	i = rand() * L + 1
	j = rand() * L + 1
	if(i .gt. L) i = i - 1
	if(j .gt. L) j = j - 1
	ip1 = i+1
	if(ip1 .gt. L) ip1 = ip1 - L
	im1 = i-1
	if(im1 .le. 0) im1 = im1 + L
	jp1 = j+1
	if(jp1 .gt. L) jp1 = jp1 - L
	jm1 = j-1
	if(jm1 .le. 0) jm1 = jm1 + L
	engy = - x1(i, j) * (x1(ip1,j)+x1(im1,j)+x1(i,jp1)+x1(i,jm1))
	if(engy .gt. 0.0) then
	x1(i, j) = - x1(i, j)
	else
	if(rand() .lt. exp(engy*2.0/T)) x1(i, j) = - x1(i, j)
	endif
440	continue
	z2 = 0.0
	do 460 k = 1, nn
	z1(k) = 0.0
	do 450 j = 1, L	
	do 450 i = 1, L
	z1(k) = z1(k) + x1(i, j) * w1((j-1)*L+i, k)
450	continue
	z1(k) = z1(k) + b1(k)
	a1(k) = 1.0/(1.0+ exp(-z1(k)))
	z2 = z2 + w2(k) * a1(k) 
460	continue
	z2 = z2 + b2
	if(z2 .gt. 0) accu = accu + 1
430	continue
	Print*, 'T/J =', T, 'Ordering probability=', accu*1.0/nval
410	continue

	end