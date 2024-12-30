string s =
"""
循环处理每一行：我们使用一个循环来处理每一行。对于每一行，我们计算出当前行的开始和结束索引，并提取出对应的子字符串。如果子字符串长度不足 n，则用空格填充
""";
int lineLength = 10;

Console.WriteLine(Reverse(s,10));

string Reverse(string s, int lineLength)
{
	int remainder = s.Length % lineLength;
	int times = (s.Length - remainder) / lineLength;
	return string.Join('\n', Enumerable.Range(0, times + 1).Select(i =>
	{
		var ce = s.Substring(i * lineLength, Math.Min(s.Length - i * lineLength, lineLength));
		if (ce.Length < lineLength) ce = $"{ce}{new string(Enumerable.Repeat(' ', lineLength - ce.Length).ToArray())}";
		return new string(ce.Reverse().ToArray());
	}));
}

/**们我：行一每理处环循
每理处来环循个一用使
我，行一每于对。行一
始开的行前当出算计们
出取提并，引索束结和
果如。串符字子的应对
n 足不度长串符字子
   充填格空用则，
**/
