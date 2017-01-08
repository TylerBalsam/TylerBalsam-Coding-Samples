// CSCI 342 - Floating Point Division
// Tyler Balsam - 4/29/15 - Dr. Metzgar
//
// Hello and welcome to my program! This program takes
// a floating point number input in hex and divides it
// without using any inherent floating point operations.
//
// There is nothing special to note about the operation
// of this program beyond the fact that it will crash if
// data is not entered in input.txt as two hex strings of
// 8 characters each separated by a space, representing
// existing floating point numbers each.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>
#include <sstream>
using namespace std;

//Globals
const long MANTISSA = 8388608;

//Functions
string chunk(string);
unsigned long divide(unsigned long, unsigned long, int&);
bool checkCode(int);
unsigned long normalize(unsigned long long, int&, int&);
unsigned long long fdivide(unsigned long long, unsigned long long);

union bitToFloat
{
	float f[sizeof(unsigned long)/sizeof(float)];
	unsigned long l;
	struct {
		unsigned int significand : 23;
		unsigned int exponent : 8;
		unsigned int sign : 1;
	} bits;
}fundend, fundsor, funres;

	void main() 
	{
		fstream fs;
		fs.open("input.txt");
		string inStr[2];
		while(fs >> inStr[0] >> inStr[1])
		{
			int code = 0;

			fundsor.l = strtoul(inStr[0].c_str(), nullptr, 16); //Converts input to ulong for union.
			fundend.l = strtoul(inStr[1].c_str(), nullptr, 16);
			funres.l = divide(fundsor.l, fundend.l, code); 
//Divides the two and returns a bitset, which is converted to u_long.
			
			if(checkCode(code)) //If there is an error,
			{
				continue; //halt the current iteration's execution.
			}

			//Else print the results
			cout << endl << chunk(inStr[0]) << " is equivalent to IEEE: " << *fundsor.f << endl;
			cout << chunk(inStr[1]) << " is equivalent to IEEE: " << *fundend.f << endl << endl;
			cout << "The quotient is ";
			printf("%1x", funres.l);
			cout << ", which is equivalent to " << *funres.f << endl << endl;
		}
		system("PAUSE");
	}

	// Author: Tyler Balsam
	// Date: 4/29/15
	//
	// Pre: The input string is vanilla, defined, and has length 8.
	//
	// Post: The string has a space after the fourth byte.
	string chunk(string inStr) 
	{
		inStr.insert(4, " ");
		return inStr;
	}


	// Author: Tyler Balsam
	// Date: 4/29/15
	//
	// Pre: indsor, indend, and code are defined.
	//
	// Post: The resultant floating point value is returned via unsigned long.
	unsigned long divide(unsigned long indsor, unsigned long indend, int& code)
	{
		//----Code Values----
		//0 - No error
		//1 - Div by 0
		//2 - Overflow
		//3 - Underflow
		//-------------------

		//Our unions for the divisor, dividend, and result, respectively.
		bitToFloat dsor, dend, result;
		dsor.l = indsor;
		dend.l = indend;
		result.l = 0;
		
		if (dsor.l == 0) //Divide by 0
		{
			code = 1;
			return false;
		}
		
		if (dend.l == 0) //0 numerator
		{
			return 0; //result is 0
		}

		//Creates output exponent.
		int exp = dsor.bits.exponent - dend.bits.exponent + 127; 
		
		//XORs the signs for the output sign.
		result.bits.sign = dsor.bits.sign ^ dend.bits.sign;

		//Create operable vars and add the implicit 1
		unsigned long long ldsorm = dsor.bits.significand + MANTISSA;
		unsigned long long ldendm = dend.bits.significand + MANTISSA;

		

		//Divides the two, normalizes the result, removes the understood 1, and sets the final significand to it.
		result.bits.significand = normalize(fdivide(ldsorm, ldendm), exp, code) - MANTISSA;
		result.bits.exponent = exp;
		
		//If there was an overflow or underflow, halt execution.
		if(code == 2 || code == 3)
			return false;
		
		
		return result.l;
	}


	// Author: Tyler Balsam
	// Date: 4/29/15
	//
	// Pre: The var code is initialized.
	//
	// Post: If an error has been detected, the function returns true and prints what the error was to the console.
	bool checkCode(int code)
	{
		switch(code)
		{
		case 0:
			return false;
			break;
		case 1:
			cout << "Error: Attempted to divide by 0." << endl << endl;
			return true;
			break;
		case 2:
			cout << "Error: An overflow has occured." << endl << endl;
			return true;
			break;
		case 3:
			cout << "Error: An underflow has occured." << endl << endl;
			return true;
			break;
		default:
			cout << "An unspecified error has occured." << endl << endl;
			return true;
			break;
		}
	}

	// Author: Tyler Balsam
	// Date: 4/29/15
	//
	// Pre: All input values are initialized.
	//
	// Post: The result of the floating point significand division is returned.
	unsigned long long fdivide(unsigned long long dsor, unsigned long long dend)
	{
		dsor <<= 23;
		unsigned long long result = dsor/dend;
		return result;
	}


	// Author: Tyler Balsam
	// Date: 4/29/15
	//
	// Pre: The input variables are defined.
	//
	// Post: The normalized significand is returned, while the exponent value is modified by reference.
	unsigned long normalize (unsigned long long inNorm, int& exp, int& code) 
	{
		if (exp < 0) //Underflow
		{
			code = 3;
			return 0;
		}

		if (exp > 255) //Overflow
		{
			code = 2;
			return false;
		}


		while(inNorm < MANTISSA)
		{
			exp--;
			inNorm <<= 1;
		}
		return inNorm;
}
