import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		
		int en = sc.nextInt();
		
		if (en < 100) {
			System.out.println("No");
		} else if (en % 100 != 0) {
			System.out.println("No");
		} else {
			System.out.println("Yes");
			
		}
	}
}