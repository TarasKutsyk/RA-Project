using System;
using System.Text;

namespace project_RA
{
    class Program
    {
        static void Main()
        {
            Console.OutputEncoding = UTF8Encoding.UTF8;
            int n = 46, m = 3;
            //Справжні значення y
            double[] y = {7.3, 7.6, 7.0, 6.6, 7.7, 7.8, 6.4, 7.5, 7.3, 7.4, 8.1, 8.4, 8.0, 7.7, 7.5,
                          6.0, 7.5, 5.8, 4.9, 5.4, 6.4, 8.0, 5.4, 6.1, 8.5, 8.5, 4.7, 6.9, 6.9, 7.8, 
                          5.1, 6.9, 7.6, 5.7, 6.5, 8.5, 7.8, 6.8, 6.8, 6.5, 7.4, 6.3, 7.8, 6.1, 6.7, 
                          6.4};
            
            //1) Лінійна модель:
            double[] y_lin = {6.72713427, 6.71477954, 6.5443019,  6.58754098, 6.99698979, 6.67123405,
                              6.87097758, 6.9633354,  7.13762521, 8.20978071, 6.99262831, 7.78408142,
                              7.63951124, 8.12425675, 6.41865551, 6.63390714, 6.93233319, 6.92026824,
                              6.88665532, 6.41659065, 6.97303819, 7.26392923, 7.35843328, 6.37865527,
                              7.35248398, 7.42637215, 6.33648144, 6.6424293,  7.16322381, 7.03480862,
                              7.02430897, 7.10840793, 6.62477897, 6.77450496, 6.68973492, 7.78944455,
                              6.93251102, 6.54303218, 6.29119716, 7.21595253, 7.17528139, 6.56682619,
                              6.67466565, 6.47400741, 6.9454602,  7.06744349};

            //2) Логарифмічна модель:
            double[] y_log = {6.87198375, 6.74806249, 6.43341664, 6.67558577, 6.66282241, 6.81692106,
                              7.06192645, 6.73466762, 6.84324101, 7.98277342, 6.88248002, 7.9761829,
                              8.47342226, 7.86421368, 6.48051447, 6.6839876,  6.80187911, 6.8005648,
                              6.7508461,  6.37882802, 6.97693326, 7.11669764, 7.41503954, 6.32183077,
                              7.41426187, 7.34677424, 6.45118345, 6.90470438, 7.09954337, 7.27524748,
                              6.74036402, 6.88611219, 6.63974107, 6.61592725, 6.69117631, 7.7300729,
                              7.01778568, 6.75456038, 6.21798562, 7.33093065, 6.98599308, 6.67781292,
                              6.83968848, 6.51015182, 6.94547145, 7.16969061};

            //3) Гіперболічна модель:
            double[] y_hyp = {7.01618779, 6.89213656, 6.63447206, 6.88525151, 6.94356648, 7.02828048,
                              7.51950116, 6.56267461, 6.76887703, 7.63370849, 6.66229544, 7.44179036,
                              8.29091941, 7.39716096, 6.58185549, 6.82486988, 6.58325267, 6.66685288,
                              6.6009512,  6.48145073, 7.06622653, 7.07953929, 7.55410156, 6.30357162,
                              7.48993345, 7.25208274, 6.59157809, 7.17729023, 6.83181428, 7.44012473,
                              6.56289648, 6.60730474, 6.70810411, 6.57038155, 6.71805557, 7.44853315,
                              7.23700926, 6.95802774, 6.29142048, 7.39405687, 6.5260465,  6.83335496,
                              7.0459673,  6.58060949, 6.9888303,  7.32708381};

            //середнє значення
            double y_ser = 0, sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum = sum + y[i];
            }
            
            y_ser = sum / n;
            
            //TSS
            double TSS = 0;
            for (int i = 0; i < n; i++)
            {
                TSS = TSS + Math.Pow((y[i] - y_ser), 2);
            }

            //ESS
            double ESS_lin = 0, ESS_log = 0, ESS_hyp = 0;
            for (int i = 0; i < n; i++)
            {
                ESS_lin = ESS_lin + Math.Pow((y[i] - y_lin[i]), 2);
                ESS_log = ESS_log + Math.Pow((y[i] - y_log[i]), 2);
                ESS_hyp = ESS_hyp + Math.Pow((y[i] - y_hyp[i]), 2);
            }

            //RSS
            double RSS_lin = 0, RSS_log = 0, RSS_hyp = 0;
            for (int i = 0; i < n; i++)
            {
                RSS_lin = RSS_lin + Math.Pow((y_lin[i] - y_ser), 2);
                RSS_log = RSS_log + Math.Pow((y_log[i] - y_ser), 2);
                RSS_hyp = RSS_hyp + Math.Pow((y_hyp[i] - y_ser), 2);
            }

            //R^2
            double R2_lin = 0, R2_log = 0, R2_hyp = 0;
            R2_lin = 1 - ESS_lin / TSS;
            R2_log = 1 - ESS_log / TSS;
            R2_hyp = 1 - ESS_hyp / TSS;

            Console.WriteLine("Коефіцієнт детермінації");
            Console.WriteLine("Лінійна модель      R2 = " + R2_lin);
            Console.WriteLine("Логарифмічна модель R2 = " + R2_log);
            Console.WriteLine("Гіперболічна модель R2 = " + R2_hyp);

            //R^2_скор
            double R2_lin_mod = 0, R2_log_mod = 0, R2_hyp_mod = 0;
            R2_lin_mod = 1 - (1 - R2_lin) * (n - 1) / (n - m - 1);
            R2_log_mod = 1 - (1 - R2_log) * (n - 1) / (n - m - 1);
            R2_hyp_mod = 1 - (1 - R2_hyp) * (n - 1) / (n - m - 1);

            Console.WriteLine("\nСкоригований коефіцієнт детермінації");
            Console.WriteLine("Лінійна модель      R2_скор = " + R2_lin_mod);
            Console.WriteLine("Логарифмічна модель R2_скор = " + R2_log_mod);
            Console.WriteLine("Гіперболічна модель R2_скор = " + R2_hyp_mod);

            //F
            double F_lin = 0, F_log = 0, F_hyp = 0;
            F_lin = (RSS_lin / ESS_lin) * ((n - m - 1) / m);
            F_log = (RSS_log / ESS_log) * ((n - m - 1) / m);
            F_hyp = (RSS_hyp / ESS_hyp) * ((n - m - 1) / m);

            Console.WriteLine("\nF-критерій Фішера");
            Console.WriteLine("Лінійна модель      F = " + F_lin);
            Console.WriteLine("Логарифмічна модель F = " + F_log);
            Console.WriteLine("Гіперболічна модель F = " + F_hyp);


        }
    }
}