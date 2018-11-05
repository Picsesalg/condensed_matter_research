        IMPLICIT NONE

cc      L is the size of 1 side of the array.
cc      ni is the number of sites.
cc      np is the number of particles.
cc      charge is the m in U_class
        INTEGER L, ni, np, itval, charge
        PARAMETER (L=30, ni=L*L, np=ni/2, itval=np/2, charge=3)
cc      x1(L, L) is the lattice with sides of size L.
cc      cp is the counter of particles.
        INTEGER x1(L, L), cp, i, j, m, n, s, t, a, b, c, d, choice

cc      distance is the distance between two positions.
cc      engy_inti_1 is the first term of the initial U_class formula.
cc      engy_init_2 is the second term of the initial U_class formula.
cc      engy_init is the overall initial energy.
        DOUBLE PRECISION distance, engy_init_1, engy_init_2, engy_init
        DOUBLE PRECISION engy_fina_1, engy_fina_2, engy_fina

cc      Initialising random numbers.\
        INTEGER*4 timearray(3)
        CALL itime(timearray)
        CALL SRAND(timearray(1) * 3600 +
     &  timearray(2) *60 + timearray(3))

        OPEN(UNIT=1, FILE="samples.csv")
1       FORMAT(I3)
2       FORMAT(A)

cc      Initalising the lattice.
        DO 10 i = 1, L
        DO 10 j = 1, L
        x1(i, j) = 0
10      CONTINUE
        cp = 0
        DO WHILE (cp < np)
        i = rand() * L + 1
        j = rand() * L + 1
        IF (x1(i, j) .EQ. 0) THEN
        x1(i, j) = 1
        cp = cp + 1
        ENDIF
        END DO

cc      Monte Carlo sampling of 100 steps to converge into the phases.
        DO 20 m = 1, 2001
        DO 30 n = 1, itval
cc      Choosing the particle to be potentially moved.
40      i = rand() * L + 1
        j = rand() * L + 1
        IF (x1(i, j) .NE. 1) GOTO 40
cc      Calculating initial energy.
        engy_init_1 = 0
        engy_init_2 = 0
        DO 50 a = 1, L
        DO 50 b = 1, L
        IF (x1(a, b) .EQ. 1) THEN
        DO 55 c = a, L
        DO 55 d = 1, L
        IF ((c .EQ. a .AND. d .GT. b) .OR. c .GT. a) THEN
        IF ((x1(c, d) .EQ. 1) .AND. (a .NE. c) .AND. (b .NE. d)) THEN
        distance = (c - a)**2 + (d - b)**2
        distance = sqrt(distance)
        engy_init_1 = engy_init_1 - log(distance)
        ENDIF
        ENDIF
55      CONTINUE
        engy_init_2 = engy_init_2 + a**2 + b**2
        ENDIF
50      CONTINUE
        engy_init_1 = engy_init_1 / 2
        engy_init_1 = charge**2 * engy_init_1
        engy_init_2 = (charge / 4) * engy_init_2
        engy_init = engy_init_1 + engy_init_2
cc      Moving the particle at (i, j) to (s, t)
cc      First movement is to move in +x by the value of m
        choice = 0
        s = i
        t = j
        DO WHILE (x1(s, t) .EQ. 1)
        choice = choice + 1
        s = i
        t = j
        SELECT CASE (choice)
        CASE (1)
        s = i + charge
        IF (s .GT. L) s = s - L
        CASE (2)
        s = i - charge
        IF (s .LT. 1) s = s + L
        CASE (3)
        t = j + charge
        IF (t .GT. L) t = t - L
        CASE (4)
        t = j - charge
        IF (t .LT. 1) t = t + L
        CASE DEFAULT
        s = rand() * L + 1
        t = rand() * L + 1
        END SELECT
        END DO
        x1(i, j) = 0
        x1(s, t) = 1
cc      Calculating the energy configuration of this new position
        engy_fina_1 = 0
        engy_fina_2 = 0
        DO 80 a = 1, L
        DO 80 b = 1, L
        IF (x1(a, b) .EQ. 1) THEN
        DO 85 c = a, L
        DO 85 d = 1, L
        IF ((c .EQ. a .AND. d .GT. b) .OR. c .GT. a) THEN
        IF ((x1(c, d) .EQ. 1) .AND. (a .NE. c) .AND. (b .NE. d)) THEN
        distance = (c - a)**2 + (d - b)**2
        distance = sqrt(distance)
        engy_fina_1 = engy_fina_1 - log(distance)
        ENDIF
        ENDIF
85      CONTINUE
        engy_fina_2 = engy_fina_2 + a**2 + b**2
        ENDIF
80      CONTINUE
cc        engy_fina_1 = engy_fina_1 / 2
        engy_fina_1 = charge**2 * engy_fina_1
        engy_fina_2 = (charge / 4) * engy_fina_2
        engy_fina = engy_fina_1 + engy_fina_2
cc      Accept flip if energy is lowered. Else, keep the original setup.
        IF (engy_fina .GT. engy_init) THEN
        x1(i, j) = 1
        x1(s, t) = 0
        ELSE IF (rand() .GT. exp(-(2 / charge) * engy_fina)) THEN
        x1(i, j) = 1
        x1(s, t) = 0
        ENDIF
30      CONTINUE
        IF (m .GT. 2000) THEN
        DO 90 a = 1, L
        DO 90 b = 1, L
        IF (b .NE. L) THEN
        WRITE(1, 1, advance="No") x1(a, b)
        WRITE(1, 2, advance="No") ","
        ELSE
        WRITE(1, 1) x1(a, b)
        ENDIF
90      CONTINUE
        ENDIF
20      CONTINUE
        END PROGRAM
