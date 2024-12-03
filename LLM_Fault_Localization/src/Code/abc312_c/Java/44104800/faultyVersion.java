import java.util.*;
import java.io.*;

public class Main{	
  public static void main(String[] args) {
		FastScanner str=new FastScanner(System.in);
		PrintWriter out=new PrintWriter(System.out);
		int n=str.nextInt();
		int m=str.nextInt();
		int[] a=new int[n];
		int[] b=new int[m];
		
		for(int i=0;i<n;i++){
			a[i]=str.nextInt();
		}
		for(int i=0;i<m;i++){
			b[i]=str.nextInt();
		}
		
    int l=0,r=1000000000;
		while(r-l>1){
			int han=(r+l)/2;
			int na=0;
			int nb=0;
			for(int i=0;i<n;i++){
        if(a[i]<=han)na++;
			}
			for(int j=0;j<m;j++){
        if(b[j]>=han)nb++;
			}
			if(na>=nb){
				r=han;
			}else{
				l=han;
			}
		}
		out.println(r);
    out.flush();
	}
}
	






	class FastScanner implements Closeable {
	private final InputStream in;
	private final byte[] buffer = new byte[1024];
	private int ptr = 0;
	private int buflen = 0;
	public FastScanner(InputStream in) {
		this.in = in;
	}
	private boolean hasNextByte() {
		if (ptr < buflen) {
			return true;
		}else{
			ptr = 0;
			try {
				buflen = in.read(buffer);
			} catch (IOException e) {
				e.printStackTrace();
			}
			if (buflen <= 0) {
				return false;
			}
		}
		return true;
	}
	private int readByte() { if (hasNextByte()) return buffer[ptr++]; else return -1;}
	private static boolean isPrintableChar(int c) { return 33 <= c && c <= 126;}
	public boolean hasNext() { while(hasNextByte() && !isPrintableChar(buffer[ptr])) ptr++; return hasNextByte();}
	public String next() {
		if (!hasNext()) throw new NoSuchElementException();
		StringBuilder sb = new StringBuilder();
		int b = readByte();
		while(isPrintableChar(b)) {
			sb.appendCodePoint(b);
			b = readByte();
		}
		return sb.toString();
	}
	public long nextLong() {
		if (!hasNext()) throw new NoSuchElementException();
		long n = 0;
		boolean minus = false;
		int b = readByte();
		if (b == '-') {
			minus = true;
			b = readByte();
		}
		if (b < '0' || '9' < b) {
			throw new NumberFormatException();
		}
		while(true){
			if ('0' <= b && b <= '9') {
				n *= 10;
				n += b - '0';
			}else if(b == -1 || !isPrintableChar(b)){
				return minus ? -n : n;
			}else{
				throw new NumberFormatException();
			}
			b = readByte();
		}
	}
	public int nextInt() {
		long nl = nextLong();
		if (nl < Integer.MIN_VALUE || nl > Integer.MAX_VALUE) throw new NumberFormatException();
		return (int) nl;
	}
	public double nextDouble() { return Double.parseDouble(next());}
	public void close() {
		try {
			in.close();
		} catch (IOException e) {
		}
	}

		}

		