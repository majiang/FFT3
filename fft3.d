module fft3;

version = working;

import std.complex : Complex, abs;
import std.traits : isFloatingPoint;
import std.range : isRandomAccessRange, ElementType;
import std.numeric : fft2p = fft, ifft2p = inverseFft;
debug import std.stdio;

/** Find inverse of x modulo m.

Warning: This implementation is correct only for {x, m} == {3, k} where k is a
power of two.
*/
size_t get_inverse(in size_t x, in size_t modulo)
in
{
    assert (x.coprime(modulo));
    assert (x == 3 || modulo == 3);
    assert (x != 1 && modulo != 1);
    assert (modulo.roundDownToPowerOf2() == modulo || x.roundDownToPowerOf2 == x);
}
out (result)
{
    assert (result * x % modulo == 1);
    version (verbose) "OK: inverse of %d modulo %d is %d".writefln(x, modulo, result);
}
body
{
    if (modulo == 3)
        return x % 3; // ternary arithmetic magic

    ulong ret = 1;
    size_t m = modulo >> 2;
    while (m)
    {
        ret *= ret;
        ret *= x;
        ret %= modulo;
        m >>= 1;
    }
    return cast(size_t)ret;
}

unittest
{
    foreach (u; [2, 4, 8, 16, 32])
    {
        u.get_inverse(3);
        3.get_inverse(u);
    }
}

/**Split radix algorithm for input with length equal two a power of two
 * multiplied by three.

Let input be x[j] and output y[k] both indexed by [0 .. N) with N = pq,
where p and q are coprime to each other.  Then DFT formula is
y[k] = sum[j] x[j] exp(-2 PI i j k / N)
and
x[j] = sum[k] y[k] exp(+2 PI i j k / N) / N.

Let p' and q' be integers with p' * p  % q = 1 and q' * q % p = 1.

If j = (Q p + P q) % N and k = (u q q' + v p p') % N then
(j k) % N = Q v p p p' + P u q q q' = P u q + Q v p
so that

y[k] = sum[j] x[j] exp(-2 PI i P[j] u[k] / p) exp(-2 PI i Q[j] v[k] / q)
= sum[P, Q] x[j[P, Q]] exp(-2 PI i P u[k] / p) exp(-2 PI i Q v[k] / q)
= sum[P] (sum[Q] x[j[P, Q]] exp(-2 PI i Q v[k] / q) ) exp(-2 PI i P u[k] / p).
*/
void fft2p3(E, R)(R range, E ret)
    if (isRandomAccessRange!R && isRandomAccessRange!E)
in
{
    assert (range.length.isPowerOfTwoTripled);
    assert (range.length == ret.length);
    immutable p = range.length.roundDownToPowerOf2 >> 1;
    immutable q = 3;
    import std.conv : text;
    assert (p * q == range.length, text(p, " * ", q, " != ", range.length));
    version (verbose) "OK: fft2p3(length = %d)".writefln(range.length);
}
body
{
    immutable p = range.length.roundDownToPowerOf2 >> 1;
    immutable q = 3;
    if (p == 1)
    {
        foreach (i, c; range.fft3())
            ret[i] = c;
        return;
    }
    immutable pp = p.get_inverse(q);
    immutable qq = q.get_inverse(p);

    auto get_j(size_t P, size_t Q)
    {
        return (Q * p + P * q) % (p * q);
    }
    auto get_k(size_t u, size_t v)
    {
        return get_j(u * qq, v * pp);
    }

    alias ElementType!E C;
    // result of first transform; q (= 3) arrays with length equal to p (= a power of two)
    C[][] x;
    foreach (Q; 0..q)
        x ~= new C[p];

    {// scope: foreach
        auto line = new C[q];
    foreach (P; 0..p)
    {
        // construct input
        foreach (Q; 0..q)
            line[Q] = range[get_j(P, Q)];
        // perform FFT of length q = 3
        line = line.fft3();
        // transpose and save
        foreach (Q; 0..q)
            x[Q][P] = line[Q];
    }
    } // line is no longer used.

    // second transform
    foreach (Q; 0..q)
        x[Q] = x[Q].fft2p();

    // rearrange
    foreach (u; 0..p)
        foreach (v; 0..q)
            ret[get_k(u, v)] = x[v][u];
}

Complex!F[] fft2p3(F = double, R)(R range) /// ditto
    if (isFloatingPoint!F && isRandomAccessRange!R)
{
    auto ret = new Complex!F[range.length];
    range.fft2p3(ret);
    return ret;
}

Complex!F[] ifft2p3(F = double, R)(R range)
    if (isFloatingPoint!F && isRandomAccessRange!R)
in
{
    assert (range.length.isPowerOfTwoTripled);
    version (verbose) "OK: ifft2p3(length = %d)".writefln(range.length);
}
body
{
    import std.math : sqrt;
    import std.algorithm : swap;
    immutable scale = (1.0 / range.length).sqrt();
    Complex!F[] ret;
    foreach (c; range)
        ret ~= c * scale;
    ret = ret.fft2p3();
    foreach (ref c; ret)
        c *= scale;
    foreach (i; 1..ret.length)
    {
        auto j = ret.length - i;
        if (j <= i)
            return ret;
        swap(ret[i], ret[j]);
    }
    assert (false);
}

unittest
{
    import std.random : uniform;
    Complex!double[] x;
    auto len = 3 << 3;
    foreach (i; 0..len)
        x ~= Complex!double(uniform(0.0, 1.0), uniform(0.0, 1.0));
    auto y = x.fft2p3();
    auto z = y.ifft2p3();
    x.writeln();
    y.writeln();
    z.writeln();
    foreach (i; 0..len)
        (x[i] - z[i]).abs().writeln();
    "OK?".writeln();
    readln();
}

size_t roundDownToPowerOf2(size_t num)
in
{
    assert (num);
}
body
{
    import core.bitop : bsr;
    return 1 << num.bsr();
}

bool isPowerOfTwoTripled(size_t num)
out (result)
{
    version (verbose) ("%d is " ~ (result ? "" : "not ") ~ "a power of 2 multiplied by 3").writefln(num);
}
body
{
    if (num == 0)
        return false;
    if (num & 1)
        return num == 3;
    return (num >> 1).isPowerOfTwoTripled();
}

alias isPowerOfTwoTripled ptt;
unittest
{
    assert (!0.ptt()); assert (!1.ptt()); assert (!2.ptt()); assert (!4.ptt()); assert (!9.ptt());
    assert (3.ptt()); assert (6.ptt()); assert (12.ptt());
}


Complex!F[] fft3(F = double, R)(R range)
    if (isFloatingPoint!F && isRandomAccessRange!R)
in
{
    import std.conv : text;
    assert (range.length == 3, text(range));
}
out (result)
{
    version (verbose) "%s -> fft3 -> %s".writefln(range, result);
}
body{
version (all)
{
/* Let range[1] = a+bi and range[2] = c+di, p = -0.5, q = -0.8....
Then ret[1] - range[0] = (a+bi)(p+qi) + (c+di)(p-qi) = (a+c)p - (b-d)q + ((a-c)q + (b+d)p)i
and ret[2] - range[0] = (a+bi)(p-qi) + (c+di)(p+qi) = (a+c)p + (b-d)q + (-(a-c)q + (b+d)p)i.
note that add.re = a+c, add.im = b+d, sub.re = a-c, sub.im = b-d.
*/
    enum re = -0.5L;
    enum im = -0.86602540378443864676372317075294L;
    auto ret = new Complex!F[3];
    foreach (ref e; ret)
        e = range[0];
    immutable add = range[1] + range[2];
    immutable sub = range[1] - range[2];
    ret[0] += add;
    immutable
        rar = re * add.re,
        isi = im * sub.im,
        rai = re * add.im,
        isr = im * sub.re;
    ret[1].re += rar - isi;    //ret[1].re += re * add.re - im * sub.im;
    ret[2].re += rar + isi;    //ret[2].re += re * add.re + im * sub.im;
    ret[1].im += rai + isr;    //ret[1].im += re * add.im + im * sub.re;
    ret[2].im += rai - isr;    //ret[2].im += re * add.im - im * sub.re;
    return ret;
}
else
{   // naive implementation.
    enum u = Complex!F(-0.5, -0.86602540378443864676372317075294);
    enum v = Complex!F(-0.5, +0.86602540378443864676372317075294);
    return [
        range[0] + range[1] + range[2],
        range[0] + range[1] * u + range[2] * v,
        range[0] + range[1] * v + range[2] * u
    ];
}}

Complex!F[] ifft3(F = double, R)(R range)
    if (isFloatingPoint!F && isRandomAccessRange!R)
in
{
    assert (range.length == 3);
}
out (result)
{
    version (verbose) "%s -> ifft3 -> %s".writefln(range, result);
}
body
{
    auto ret = [range[0], range[2], range[1]].fft3();
    foreach (ref e; ret)
        e *= 0x0.5555555555555555p+0;
    return ret;
}

unittest
{
    import std.random;
    import std.stdio;
    Complex!double[]  x;
    foreach (i; 0..3)
        x ~= Complex!double(uniform(0.0, 1.0), uniform(0.0, 1.0));
    x.fft3().ifft3();
}


bool coprime(size_t x, size_t y)
out (result)
{
    version (verbose) ("%d " ~ (result ? "is " : "not ") ~ "coprime %d").writefln(x, y);
}
body
{
    if (x == 1 || y == 1)
        return true;
    if (x & y & 1)
        return (x < y ? x.coprime(y - x) : y.coprime(x - y));
    if (~x & ~y & 1)
        return false;
    if (!x || !y)
        return false;
    if (x & 1)
        return x.coprime(y >> 1);
    assert (y & 1);
        return (x >> 1).coprime(y);
}

unittest
{
    assert (3.coprime(5));
    assert (13.coprime(5));
    assert (13.coprime(35));
    assert (!4.coprime(6));
    assert (!3.coprime(15));
    assert (!30.coprime(25));
    assert (1.coprime(5));
}
