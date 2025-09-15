import os
import csv
from datetime import datetime, timedelta
# 打开文件并读取内容
def read_special_lines(file_path):
    with open('./electricity_2012.txt', 'a+', encoding='utf-8') as fout:
        with open('./LD2011_2014.txt', 'r', encoding='utf-8') as fin:
            content = fin.readlines()
            for line in content:
                date = line.split(";")[0].split(" ")[0].split("-")[0].split("\"")[1]
                if date == '2012':
                    fout.write(line)
        fin.close()
    fout.close()

def construct_csv():
    with open('/Users/lzd/Documents/electricity/electricity_2012_origin.txt', 'r', encoding='utf-8') as fin:
        content = fin.readlines()

    title = "date,MT_001,MT_002,MT_003,MT_004,MT_005,MT_006,MT_007,MT_008,MT_009,MT_010,MT_011,MT_013,MT_014,MT_016,MT_017,MT_018,MT_019,MT_020,MT_021,MT_022,MT_023,MT_025,MT_026,MT_027,MT_028,MT_029,MT_031,MT_034,MT_035,MT_036,MT_037,MT_038,MT_040,MT_042,MT_043,MT_044,MT_045,MT_046,MT_047,MT_048,MT_049,MT_050,MT_051,MT_052,MT_053,MT_054,MT_055,MT_056,MT_057,MT_058,MT_059,MT_060,MT_061,MT_062,MT_063,MT_064,MT_065,MT_066,MT_067,MT_068,MT_069,MT_070,MT_071,MT_072,MT_073,MT_074,MT_075,MT_076,MT_077,MT_078,MT_079,MT_080,MT_081,MT_082,MT_083,MT_084,MT_085,MT_086,MT_087,MT_088,MT_089,MT_090,MT_091,MT_093,MT_094,MT_095,MT_096,MT_097,MT_098,MT_099,MT_100,MT_101,MT_102,MT_103,MT_104,MT_105,MT_114,MT_118,MT_119,MT_123,MT_124,MT_125,MT_126,MT_128,MT_129,MT_130,MT_131,MT_132,MT_135,MT_136,MT_137,MT_138,MT_139,MT_140,MT_141,MT_142,MT_143,MT_145,MT_146,MT_147,MT_148,MT_149,MT_150,MT_151,MT_153,MT_154,MT_155,MT_156,MT_157,MT_158,MT_159,MT_161,MT_162,MT_163,MT_164,MT_166,MT_168,MT_169,MT_171,MT_172,MT_174,MT_175,MT_176,MT_180,MT_182,MT_183,MT_187,MT_188,MT_189,MT_190,MT_191,MT_192,MT_193,MT_194,MT_195,MT_196,MT_197,MT_198,MT_199,MT_200,MT_201,MT_202,MT_203,MT_204,MT_205,MT_206,MT_207,MT_208,MT_209,MT_210,MT_211,MT_212,MT_213,MT_214,MT_215,MT_216,MT_217,MT_218,MT_219,MT_220,MT_221,MT_222,MT_223,MT_225,MT_226,MT_227,MT_228,MT_229,MT_230,MT_231,MT_232,MT_233,MT_234,MT_235,MT_236,MT_237,MT_238,MT_239,MT_240,MT_241,MT_242,MT_243,MT_244,MT_245,MT_246,MT_247,MT_248,MT_249,MT_250,MT_251,MT_252,MT_253,MT_254,MT_256,MT_257,MT_258,MT_259,MT_260,MT_261,MT_262,MT_263,MT_264,MT_265,MT_266,MT_267,MT_268,MT_269,MT_270,MT_271,MT_272,MT_273,MT_274,MT_275,MT_276,MT_277,MT_278,MT_279,MT_280,MT_281,MT_282,MT_283,MT_284,MT_285,MT_286,MT_287,MT_288,MT_290,MT_291,MT_292,MT_293,MT_294,MT_295,MT_296,MT_297,MT_298,MT_299,MT_300,MT_301,MT_302,MT_303,MT_304,MT_306,MT_307,MT_309,MT_310,MT_311,MT_312,MT_313,MT_314,MT_315,MT_316,MT_317,MT_318,MT_319,MT_320,MT_321,MT_323,MT_324,MT_325,MT_326,MT_327,MT_328,MT_329,MT_330,MT_331,MT_333,MT_334,MT_335,MT_336,MT_338,MT_339,MT_340,MT_341,MT_342,MT_343,MT_344,MT_345,MT_346,MT_347,MT_348,MT_349,MT_350,MT_351,MT_352,MT_353,MT_354,MT_355,MT_356,MT_357,MT_358,MT_359,MT_360,MT_361,MT_362,MT_363,MT_364,MT_365,MT_366,MT_367,MT_368,MT_369"
    # 打开输出CSV文件
    with open('/Users/lzd/Documents/electricity/electricity_2012_origin.csv', 'a+', newline='', encoding='utf-8') as fout:
        csv_writer = csv.writer(fout, delimiter=',')
        csv_writer.writerow(title.split(","))
        for line in content:
            # 去除行末的换行符
            line = line.strip()
            # 将逗号替换为点
            line = line.replace(',', '.')
            # 按照分号分割每一列
            columns = line.split(';')

            columns[0] = columns[0].split("\"")[1]
            # 检查日期是否为2012年
            # date = columns[0].split(" ")[0].split("-")[0].strip('"')
            # if date == '2012':
            csv_writer.writerow(columns)

def calcute_hour():
    with open('/Users/lzd/Documents/electricity/electricity_2012_origin.csv', 'r', encoding='utf-8') as fout:
        csv_reader = csv.reader(fout, delimiter=',')
        
        with open('/Users/lzd/Documents/electricity/electricity_2012_hour.csv', 'a+', newline='', encoding='utf-8') as fout_hour:
            csv_writer = csv.writer(fout_hour, delimiter=',')

            # 读取并写入标题行
            headers = next(csv_reader, None)
            if headers is None:
                print(f"文件为空或没有标题行。")
                return
            csv_writer.writerow(headers)

            # 初始化变量
            next_hour_time = None
            current_data = None
            index = [1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,21,22,23,25,26,27,28,29,31,34,35,36,37,38,40,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,93,94,95,96,97,98,99,100,101,102,103,104,105,114,118,119,123,124,125,126,128,129,130,131,132,135,136,137,138,139,140,141,142,143,145,147,148,149,150,151,153,154,155,156,157,158,159,161,162,163,164,166,168,169,171,172,174,175,176,180,182,183,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,306,307,309,310,311,312,313,314,315,316,317,318,319,320,321,323,324,325,326,327,328,329,330,331,333,334,335,336,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369]
            
            for row in csv_reader:
                current_time_str = row[0]
                current_time = datetime.strptime(current_time_str, '%Y-%m-%d %H:%M:%S')
                if next_hour_time is None:
                    next_hour_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

                if current_time < next_hour_time:
                    if current_data is None:
                            current_data = [float(row[i]) for i in index]
                    # 累加当前行的数据到current_data
                    else:
                        j = 0
                        for i in index:
                            current_data[j] += float(row[i])
                            j += 1
                else:
                    j = 0
                    for i in index:
                        current_data[j] += float(row[i])
                        j += 1
                    # 写入current_data到输出文件
                    csv_writer.writerow([current_time_str] + current_data)

                    # 更新current_hour和current_data
                    next_hour_time = None
                    current_data = None

calcute_hour()