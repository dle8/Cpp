/*
    Fast input:
        ios_base::sync_with_stdio(false);
            This disables the synchronization between the C and C++ standard streams. By default, all 
            standard streams are synchronized, which in practice allows you to mix C- and C++-style I/O 
            and get sensible and expected results. If you disable the synchronization, then C++ streams 
            are allowed to have their own independent buffers, which makes mixing C- and C++-style I/O 
            an adventure.

            Also keep in mind that synchronized C++ streams are thread-safe (output from different threads 
            may interleave, but you get no data races).

        cin.tie(NULL):
            This unties cin from cout. Tied streams ensure that one stream is flushed automatically before 
            each I/O operation on the other stream.

            By default cin is tied to cout to ensure a sensible user interaction. For example:

            std::cout << "Enter name:";
            std::cin >> name;
            If cin and cout are tied, you can expect the output to be flushed (i.e., visible on the console) 
            before the program prompts input from the user. If you untie the streams, the program might block 
            waiting for the user to enter their name but the "Enter name" message is not yet visible (because 
            cout is buffered by default, output is flushed/displayed on the console only on demand or when 
            the buffer is full).

            So if you untie cin from cout, you must make sure to flush cout manually every time you want to 
            display something before expecting input on cin.

            In conclusion, know what each of them does, understand the consequences, and then decide if you 
            really want or need the possible side effect of speed improvement.
    
    Fast Output:
        In general, it may be faster to store the answer all in a single string (C++) or 
        StringBuffer (Java) and outputting it with a single function call. This method avoids the 
        overhead of calling an output method many times, especially if the output is generated in many 
        parts.

        The CF blog mentioned above notes that when printing many lines in C++, it may be faster to 
        use the newline character \n in place of endl. Output streams in C++ (such as cout and 
        ofstream) are buffered, meaning that they don't immediately print their output, but store 
        some of it. At some point, the buffer's contents are written (i.e. "flushed") to the output 
        device (e.g the standard output stream or a file). Buffering the output helps with efficiency 
        if accessing the output device (like a file) is slow. Because endl flushes the output, it may 
        be faster to use \n instead and avoid unnecessary flushes.

*/

/**
 * Description: Fast input and output.
 * Time: input is $\sim$300ms faster for $10^6$ long longs on CF
 * Source: 
 	* https://codeforces.com/gym/102394/submission/64154785
 	* https://codeforces.com/contest/1254/submission/65420506 (neal)
 	* https://codeforces.com/blog/entry/45835 (AI.Cash)
 * Verification: https://codeforces.com/gym/102394/problem/G 
 */

namespace FastIO {
	const int BSZ = 1<<15; ////// INPUT
	char ibuf[BSZ]; int ipos, ilen;
	char nc() { // next char
		if (ipos == ilen) {
			ipos = 0; ilen = fread(ibuf,1,BSZ,stdin);
			if (!ilen) return EOF;
		}
		return ibuf[ipos++];
	}
	void rs(str& x) { // read str
		char ch; while (isspace(ch = nc()));
		do { x += ch; } while (!isspace(ch = nc()) && ch != EOF);
	}
	template<class T> void ri(T& x) { // read int or ll
		char ch; int sgn = 1;
		while (!isdigit(ch = nc())) if (ch == '-') sgn *= -1;
		x = ch-'0'; while (isdigit(ch = nc())) x = x*10+(ch-'0');
		x *= sgn;
	}
	template<class T, class... Ts> void ri(T& t, Ts&... ts) { 
		ri(t); ri(ts...); } // read ints
	////// OUTPUT (call initO() at start)
	char obuf[BSZ], numBuf[100]; int opos;
	void flushOut() { fwrite(obuf,1,opos,stdout); opos = 0; }
	void wc(char c) { // write char
		if (opos == BSZ) flushOut();
		obuf[opos++] = c; }
	void ws(str s) { trav(c,s) wc(c); } // write str
	template<class T> void wi(T x, char after = '\0') { /// write int
		if (x < 0) wc('-'), x *= -1;
		int len = 0; for (;x>=10;x/=10) numBuf[len++] = '0'+(x%10);
		wc('0'+x); R0F(i,len) wc(numBuf[i]);
		if (after) wc(after);
	}
	void initO() { assert(atexit(flushOut) == 0); } /// auto-flush output
}
/// using namespace FastIO;
/// initO(); int a,b; ri(a,b); wi(b,'\n'); wi(a,'\n'); 