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

	open(unit=1, file="samples.csv")
cc	open(unit=2, file="sample2.csv")
	do 140 i = 1, 900
	write(1, 20, advance="No") i
	write(1, 15, advance="No") ","
140	continue
	write(1, 15) "Label"
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
	x1(i, j) = x1(i, j)*info
	write(1, 10, advance="No") x1(i, j)
10	format(I2)
	write(1, 15, advance="No") ","
15	format(A)
180	continue
	write(1, 20) 1
20	format(I3)
	do 190 i = 1, L
	do 190 j = 1, L
	x2(i, j) = x2(i, j)*info
	write(1, 10, advance="No") x2(i, j)
	write(1, 15, advance="No") ","
190	continue
	write(1, 20) 0
	endif
160	continue
	close(1)

	open(unit=2, file="temperatures.csv")
	do 210 i = 1, 900
	write(2, 20, advance="No") i
	write(2, 15, advance="No") ","
210	continue
	write(2, 15) "Label"
	
	do 410 n = 0, 75
	T = 3.5 - n * 0.025
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
440     continue
	do 441 i = 1, L
	do 441 j = 1, L
	write(2, 10, advance="No") x1(i, j)
	write(2, 15, advance="No") ","
441	continue
	write(2, 20) 1
430     continue
410     continue


	close(2)
	end program
