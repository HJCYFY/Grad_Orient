#include <emmintrin.h>
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <fstream>

using namespace std;
// #define DEBUG_SHOW
// #define DEBUG_FOUT
// #define DEBUG_COUT

#define PI 3.14159265f

void ucharImg2shortImgAlign16SSE(uchar *in ,unsigned int width ,unsigned int height ,unsigned int channels ,int16_t *out)
{
    long long unsigned int Area = width * height * channels;  
    const unsigned int leave = Area%16;
    Area -= leave;    
    __m128i ucharData,lower8Byte,upper8Byte;
    __m128i zeros = _mm_setzero_si128(); 
    for(long long unsigned int i=0; i<Area ;)
    {
        ucharData =  _mm_load_si128((__m128i *)in); 
        lower8Byte = _mm_unpacklo_epi8(ucharData, zeros); 
        upper8Byte = _mm_unpackhi_epi8(ucharData, zeros); 
        _mm_store_si128((__m128i*)out,lower8Byte);
        out+=8;
        _mm_store_si128((__m128i*)out,upper8Byte);
        out+=8;
        i+=16;
        in+=16;
    }
    for(long long unsigned int i=0; i<leave ;++i)
    {
        *out = *in;
        ++in;   ++out;
    }
}
void ucharImg2shortImgSSE(uchar *in ,unsigned int width ,unsigned int height ,unsigned int channels  ,int16_t *out)
{
    long long unsigned int Area = width * height * channels;  
    const unsigned int leave = Area%16;
    Area -= leave;    
    __m128i ucharData,lower8Byte,upper8Byte;
    __m128i zeros = _mm_setzero_si128(); 
    for(long long unsigned int i=0; i<Area ;)
    {
        ucharData =  _mm_loadu_si128((__m128i*)in); 
        lower8Byte = _mm_unpacklo_epi8(ucharData, zeros); 
        upper8Byte = _mm_unpackhi_epi8(ucharData, zeros); 
        _mm_storeu_si128((__m128i*)out,lower8Byte);
        out+=8;
        _mm_storeu_si128((__m128i*)out,upper8Byte);
        out+=8;
        i+=16;
        in+=16;
    }
    for(long long unsigned int i=0; i<leave ;++i)
    {
        *out = *in;
        ++in;   ++out;
    }
}
void ucharImg2shortImg(uchar *in ,unsigned int width ,unsigned int height ,unsigned int channels  ,int16_t *out)
{
    const long long unsigned int Area = width * height * channels;
    for(long long unsigned int i=0; i<Area ; ++i)
    {
        out[i] = in[i];
    }
}

// 速度慢
void ucharImg2floatImgSSE(uchar *in ,unsigned int width ,unsigned int height ,unsigned int channels  ,float *out)
{
    long long unsigned int Area = width * height * channels;  
    const unsigned int leave = Area%16;
    Area -= leave;    
    __m128i ucharData;
    __m64 eightuInt8;
    __m128 fourFloat; 
    for(long long unsigned int i=0; i<Area ;)
    {
        // 加载16个 uint8
        ucharData =  _mm_loadu_si128((__m128i*)in); 
        // 转换成8个 uint8
        eightuInt8 = _mm_movepi64_pi64(ucharData);
        // 转换成4个 float32
        fourFloat = _mm_cvtpu8_ps(eightuInt8); 
        // 存储
        _mm_storeu_ps(out, fourFloat); 

        out+=4;
        // 移位四字节
        ucharData = _mm_srli_si128(ucharData, 4); 
        // 转换成8个 uint8
        eightuInt8 = _mm_movepi64_pi64(ucharData);
        // 转换成4个 float32
        fourFloat = _mm_cvtpu8_ps(eightuInt8);
        // 存储
        _mm_storeu_ps(out, fourFloat); 

        out+=4;
        // 移位四字节
        ucharData = _mm_srli_si128(ucharData, 4); 
        // 转换成8个 uint8
        eightuInt8 = _mm_movepi64_pi64(ucharData);
        // 转换成4个 float32
        fourFloat = _mm_cvtpu8_ps(eightuInt8);
        // 存储
        _mm_storeu_ps(out, fourFloat); 

        out+=4;
        // 移位四字节
        ucharData = _mm_srli_si128(ucharData, 4); 
        // 转换成8个 uint8
        eightuInt8 = _mm_movepi64_pi64(ucharData);
        // 转换成4个 float32
        fourFloat = _mm_cvtpu8_ps(eightuInt8);
        // 存储
        _mm_storeu_ps(out, fourFloat); 
        out+=4;

        // __m64 four16Intu = _mm_movepi64_pi64(ucharData);
        // __m128 four8uintll = _mm_cvtpu8_ps(__m64 a); 

        // __m128 four8uintll = _mm_cvtpu8_ps(__m64 a); 
        // // __m128 four32Floatl = _mm_cvtpi16_ps(four16Intl); 
        // // __m128 four32Floatu = _mm_cvtpi16_ps(four16Intu); 
        // _mm_store_ps(out, four16Intl); 
        // out+=4;
        // _mm_store_ps(out, four16Intl); 
        // out+=4;
        in+=16;
        i+=16;
    }
    for(long long unsigned int i=0; i<leave ;++i)
    {
        *out = *in;
        ++in;   ++out;
    }
}
// 速度快
void ucharImg2floatImgSSE2(uchar *in ,unsigned int width ,unsigned int height ,unsigned int channels  ,float *out)
{
    long long unsigned int Area = width * height * channels;  
    const unsigned int leave = Area%16;
    Area -= leave;    
    __m128i ucharData,lowerUint16,upperUint16;
    __m128i llowerUint32,lupperUint32,ulowerUint32,uupperUint32;
    __m128 llowerFloat,lupperFloat,ulowerFloat,uupperFloat; 
    __m128i zeros = _mm_setzero_si128(); 
    for(long long unsigned int i=0; i<Area ;)
    {
        // 加载16个 uint8
        ucharData =  _mm_loadu_si128((__m128i*)in);
        // 转 uint16 
        lowerUint16 = _mm_unpacklo_epi8(ucharData, zeros); 
        upperUint16= _mm_unpackhi_epi8(ucharData, zeros); 
        // 转 uint32
        llowerUint32 = _mm_unpackhi_epi16(zeros, lowerUint16); 
        llowerUint32 = _mm_srai_epi32(llowerUint32, 16); 
        lupperUint32 = _mm_unpackhi_epi16(zeros, lowerUint16); 
        lupperUint32 = _mm_srai_epi32(lupperUint32, 16); 
        ulowerUint32 = _mm_unpackhi_epi16(zeros, upperUint16); 
        ulowerUint32 = _mm_srai_epi32(ulowerUint32, 16); 
        uupperUint32 = _mm_unpackhi_epi16(zeros, upperUint16); 
        uupperUint32 = _mm_srai_epi32(uupperUint32, 16); 

        // 转 float
        llowerFloat = _mm_cvtepi32_ps(llowerUint32); 
        lupperFloat = _mm_cvtepi32_ps(lupperUint32); 
        ulowerFloat = _mm_cvtepi32_ps(ulowerUint32); 
        uupperFloat = _mm_cvtepi32_ps(uupperUint32);

        _mm_storeu_ps(out, llowerFloat); 
        out+=4;
        _mm_storeu_ps(out, lupperFloat); 
        out+=4;
        _mm_storeu_ps(out, ulowerFloat); 
        out+=4;
        _mm_storeu_ps(out, uupperFloat); 
        out+=4;    
        in+=16;
        i+=16;  
    }
    for(long long unsigned int i=0; i<leave ;++i)
    {
        *out = *in;
        ++in;   ++out;
    }
}

void ucharImg2floatImg(uchar *in ,unsigned int width ,unsigned int height ,unsigned int channels  ,float *out)
{
    const long long unsigned int Area = width * height * channels;
    for(long long unsigned int i=0; i<Area ; ++i)
    {
        out[i] = in[i];
    }
}

void CalGrad(cv::Mat& img , float* Grad ,float* Orient, bool full)
{
    assert(img.channels() == 1);
    const long unsigned int width = img.cols,height = img.rows;
    assert(width*height < 4.29e9);

    const long unsigned int aliquot = (width * height)/16;  
    const long unsigned int mod = (width * height)%16;
    __m128i ucharData,lower8Byte,upper8Byte;
    __m128i zeros = _mm_setzero_si128();

    #ifdef DEBUG_SHOW
        cv::Mat DebugImg(img.rows,img.cols,CV_8UC1);
        uchar* data = DebugImg.data;
    #endif
    
    // 将图像转为 int16类型
    int16_t* __attribute__((aligned(16))) Int16Img = new int16_t[width*height];
    int16_t* out = Int16Img;
    uchar* in = img.data;
    for(long unsigned int i=0; i<aliquot; ++i)
    {
        ucharData =  _mm_load_si128((__m128i *)in); 
        lower8Byte = _mm_unpacklo_epi8(ucharData, zeros); 
        upper8Byte = _mm_unpackhi_epi8(ucharData, zeros); 
        _mm_store_si128((__m128i*)out,lower8Byte);
        out+=8;
        _mm_store_si128((__m128i*)out,upper8Byte);
        out+=8;
        in+=16;
    }
    for(long unsigned int i=0; i<mod ;++i)
    {
        *out = *in;
        ++in;   ++out;
    }

    #ifdef DEBUG_SHOW
        unsigned int area = img.rows*img.cols;

        int16_t* Int16Data = Int16Img;
        for(int i=0;i<area;++i) 
        {
            data[i] = Int16Data[i];
        }
        imshow("Int16",DebugImg);
        cv::waitKey(0);
    #endif
    #ifdef DEBUG_COUT
        cout<<" Int16 "<<endl;
        for(int i=0;i<height;++i)
        {
            for(int j=0;j<width;++j)
            {
                cout<<Int16Img[i*width+j]<<" ";
            }
            cout<<endl;
        }
    #endif

    //求梯度
    int16_t* __attribute__((aligned(16))) Gy = new int16_t[width*height];
    int16_t* __attribute__((aligned(16))) Gx = new int16_t[width*height];

    // printf("align %ld\n",__alignof__(Gy));

    unsigned long int row,col;
    const unsigned long int aliqW = width/8;
    const unsigned long int modW = width%8;
    __m128i R08Int16,R18Int16,Result;
    __m128i Two = _mm_set1_epi16(2);
    {   /*   y 方向梯度   */
        int16_t* row0Data ;
        int16_t* row1Data ;
        int16_t* out; 
        {   /*  row 0  */
            row0Data = Int16Img;
            row1Data = Int16Img+width;
            out = Gy;
            for(col=0;col<aliqW;++col)
            {
                R08Int16 = _mm_loadu_si128((__m128i*)row0Data); 
                R18Int16 = _mm_loadu_si128((__m128i*)row1Data);     
                Result = _mm_subs_epi16(R18Int16,R08Int16);
                Result = _mm_mullo_epi16(Result,Two);                 
                _mm_storeu_si128((__m128i*)out, Result); 
                row0Data+=8;
                row1Data+=8;
                out+=8;
            }
            for(col=0;col<modW;++col)
            {
                *out = (*row1Data-*row0Data)*2;
                ++out;  ++row1Data;   ++row0Data;
            }
        }
        {   /*  row h-1  */
            row0Data = Int16Img+(height-2)*width;
            row1Data = Int16Img+(height-1)*width;
            out = Gy+(height-1)*width;

            for(col=0;col<aliqW;++col)
            {
                R08Int16 = _mm_loadu_si128((__m128i*)row0Data); 
                R18Int16 = _mm_loadu_si128((__m128i*)row1Data);
                Result = _mm_subs_epi16(R18Int16,R08Int16);
                Result = _mm_mullo_epi16(Result,Two);      
                _mm_storeu_si128((__m128i*)out, Result); 
                row0Data+=8;
                row1Data+=8;
                out+=8;
            }
            for(col=0;col<modW;++col)
            {
                *out = (*row1Data-*row0Data)*2;
                ++out;  ++row1Data;   ++row0Data;
            }
        }
        {   /* row 1 to row h-1  */
            row0Data = Int16Img;
            row1Data = Int16Img+2*width;
            out = Gy+width;
            const long unsigned int aliqM = width*(height-2)/8;
            const long unsigned int modM = width*(height-2)%8;

            for(row=0;row<aliqM;++row)
            {
                R08Int16 = _mm_loadu_si128((__m128i*)row0Data); 
                R18Int16 = _mm_loadu_si128((__m128i*)row1Data);     
                Result = _mm_subs_epi16(R18Int16,R08Int16);              
                _mm_storeu_si128((__m128i*)out, Result); 
                row0Data+=8;
                row1Data+=8;
                out+=8;
            }
            for(col=0;col<modM;++col)
            {
                *out = (*row1Data-*row0Data);
                ++out;  ++row1Data;   ++row0Data;
            }
        }
        // // 输出 y 方向梯度
        // for(int i=0;i<height;++i)
        // {
        //     for(int j=0;j<width;++j)
        //     {
        //         cout<<Gy[i*width+j]<<" ";
        //     }
        //     cout<<endl;
        // }
        // cout<<endl;
    }

    #ifdef DEBUG_SHOW
        Int16Data = Gy;
        for(int i=0;i<area;++i) 
        {
            data[i] = abs(Int16Data[i]/2.0);
        }
        imshow("Gy",DebugImg);
        cv::waitKey(0);
    #endif
    #ifdef DEBUG_COUT
        cout<<" Gy "<<endl;
        for(int i=0;i<height;++i)
        {
            for(int j=0;j<width;++j)
            {
                cout<<Gy[i*width+j]<<" ";
            }
            cout<<endl;
        }
    #endif

    {   /*  x 方向梯度  */  
        int16_t* col0Data ;
        int16_t* col1Data ;
        int16_t* out; 
        {   //  col 0   
            for(row=0;row<height;++row)
            {
                col0Data = Int16Img+row*width;
                col1Data = col0Data+1;
                out = Gx+row*width;
                *out = (*col1Data - *col0Data)*2;                
            }
        }
        {   //  col w-1   
            for(row=0;row<height;++row)
            {
                col0Data = Int16Img+(row+1)*width-2;
                col1Data = col0Data+1;
                out = Gx+(row+1)*width-1;
                *out = (*col1Data - *col0Data)*2;                
            }
        }
        {   //  col 1 to  w-1   

            const long unsigned int aliqM = (width-2)/8;
            const long unsigned int modM = (width-2)%8;
            for(row=0;row<height;++row)
            {                
                col0Data = Int16Img+row*width;
                col1Data = col0Data+2; 
                out = Gx+row*width+1;
                for(col=0;col<aliqM;++col)    
                {                    
                    R08Int16 = _mm_loadu_si128((__m128i*)col0Data); 
                    R18Int16 = _mm_loadu_si128((__m128i*)col1Data);    
                    Result = _mm_subs_epi16(R18Int16,R08Int16);            
                    _mm_storeu_si128((__m128i*)out, Result); 
                    col0Data+=8;
                    col1Data+=8;
                    out+=8;
                }                     
                for(col=0;col<modM;++col)
                {
                    *out = (*col1Data-*col0Data);
                    ++out;  ++col1Data;   ++col0Data;
                }   
            }
        }             
        // // 输出 x 方向梯度
        // for(int i=0;i<height;++i)
        // {
        //     for(int j=0;j<width;++j)
        //     {
        //         cout<<Gx[i*width+j]<<" ";
        //     }
        //     cout<<endl;
        // }   
        // cout<<endl;
    }

    #ifdef DEBUG_SHOW
        Int16Data = Gx;
        for(int i=0;i<area;++i) 
        {
            data[i] = abs(Int16Data[i]/2.0);;
        }
        imshow("Gx",DebugImg);
        cv::waitKey(0);
    #endif
    #ifdef DEBUG_COUT
        cout<<" Gx "<<endl;
        for(int i=0;i<height;++i)
        {
            for(int j=0;j<width;++j)
            {
                cout<<Gx[i*width+j]<<" ";
            }
            cout<<endl;
        }
    #endif

    delete[] Int16Img;

    {   /*  计算 Grad 和  Orient   */

        const long int n=1000;
        const long int b=1;
        float* acos_table = new float[2*(n+b)];
        float *a1 = acos_table+n+b;
        int i;
        for( i=-n-b; i<-n; i++ )   a1[i]=PI;
        for( i=-n; i<n; i++ )      a1[i]=acos(i/float(n));
        for( i=n; i<n+b; i++ )     a1[i]=0;
        int16_t* _Gx =  Gx;
        int16_t* _Gy =  Gy;
        float* _Grad = Grad;
        float* _Orient = Orient;
        unsigned long int index;
        const unsigned long int aliqA = (width*height)/8;
        const unsigned long int modA = (width*height)%8;
        __m128i GxData,GyData,GxInt32,GyInt32;
        __m128 GxFloat,GyFloat,Gradient;
        __m128 half = _mm_set1_ps(0.5);
        __m128 max = _mm_set1_ps(1e10f);
        __m128 num = _mm_set1_ps(1e3f);
        float ori[4];

        for(index=0;index<aliqA;++index)
        {
            GxData = _mm_load_si128((__m128i*)_Gx); 
            GyData = _mm_load_si128((__m128i*)_Gy); 
            // 低四位 转 int32
            GxInt32 = _mm_unpacklo_epi16(zeros, GxData);
            GxInt32 = _mm_srai_epi32(GxInt32, 16);
            GyInt32 = _mm_unpacklo_epi16(zeros, GyData);
            GyInt32 = _mm_srai_epi32(GyInt32, 16);
            // 转 float            
            GxFloat = _mm_mul_ps(_mm_cvtepi32_ps(GxInt32),half);  
            GyFloat = _mm_mul_ps(_mm_cvtepi32_ps(GyInt32),half);

            // 计算 幅度 平方和再开方
            Gradient=_mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(GxFloat,GxFloat),_mm_mul_ps(GyFloat,GyFloat)));
            // 存储
            _mm_store_ps(_Grad,Gradient);
            _Grad+=4;
            // 梯度求倒数
            Gradient = _mm_min_ps(_mm_rcp_ps(Gradient),max);            
            // 方向归一化到 -1000 ～ 1000
            GxFloat = _mm_mul_ps( _mm_mul_ps(GxFloat,Gradient), num );
            //存储到ori
            _mm_storeu_ps(ori,GxFloat);


            for(int i=0;i<4;++i)
            {
                *_Orient = a1[(int)ori[i]];
                ++_Orient;
            }
                  
            // 高四位 转 int32
            GxInt32 = _mm_unpackhi_epi16(zeros, GxData);
            GxInt32 = _mm_srai_epi32(GxInt32, 16);
            GyInt32 = _mm_unpackhi_epi16(zeros, GyData);
            GyInt32 = _mm_srai_epi32(GyInt32, 16);
            // 转 float            
            GxFloat = _mm_mul_ps(_mm_cvtepi32_ps(GxInt32),half);  
            GyFloat = _mm_mul_ps(_mm_cvtepi32_ps(GyInt32),half);  
 
            // 计算 幅度 平方和再开方
            Gradient=_mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(GxFloat,GxFloat),_mm_mul_ps(GyFloat,GyFloat)));
            _mm_store_ps(_Grad,Gradient);
            _Grad+=4;
            // 梯度求倒数
            Gradient = _mm_min_ps(_mm_rcp_ps(Gradient),max);            
            // 方向归一化到 -1000 ～ 1000
            GxFloat = _mm_mul_ps( _mm_mul_ps(GxFloat,Gradient), num );
            //存储到ori
            _mm_storeu_ps(ori,GxFloat);

            for(int i=0;i<4;++i)
            {
                *_Orient = a1[(int)ori[i]];
                ++_Orient;
            }
            _Gx+=8;
            _Gy+=8;
        }
        for(index=0;index<modA;++index)
        {
            *_Grad = sqrt((*_Gx)*(*_Gx)+(*_Gy)*(*_Gy))*0.5;

            *_Orient = a1[(int)(((*_Gx)*min(1.0f/(*_Grad),1e10f))*1000.f)];
            ++_Gx;  ++_Gy;  ++_Grad;    //++_Orient;
        }   
        delete[] acos_table; 
    }  

    #ifdef DEBUG_SHOW
        float* FloatData = Grad;
        for(int i=0;i<area;++i) 
        {
            data[i] = FloatData[i];
        }
        imshow("Grad",DebugImg);
        cv::waitKey(0);
    #endif
    #ifdef DEBUG_COUT
        cout<<" Grad "<<endl;
        for(int i=0;i<height;++i)
        {
            for(int j=0;j<width;++j)
            {
                cout<<Grad[i*width+j]<<" ";
            }
            cout<<endl;
        }
    #endif
    if(full)
    {
        int16_t* _Gy =  Gy;
        float* _Orinet = Orient;
        __m128i GyData, GyInt32;
        __m128 Orientation;
        const unsigned long int aliqA = (width*height)/8;
        const unsigned long int modA = (width*height)%8;

        __m128i _m128i_Zero = _mm_set1_epi32(0);
        __m128 _m128_PI = _mm_set1_ps(PI);

        unsigned long int index;
        for(index=0;index<aliqA;++index)
        {
            GyData = _mm_load_si128((__m128i*)_Gy); 
            GyInt32 = _mm_unpacklo_epi16(zeros, GyData);
            GyInt32 = _mm_srai_epi32(GyInt32, 16);
            // 这里 (PI～3/2PI) 与 (3/2PI～2PI)是调换过来的 ，对编码应该没有影响
            Orientation = _mm_add_ps( _mm_load_ps(_Orinet), _mm_and_ps( _mm_castsi128_ps(_mm_cmplt_epi32(GyInt32,_m128i_Zero)) ,_m128_PI) );
            _mm_store_ps(_Orinet,Orientation);

            _Orinet+=4;
            GyInt32 = _mm_unpackhi_epi16(zeros, GyData);
            GyInt32 = _mm_srai_epi32(GyInt32, 16);
            Orientation = _mm_add_ps( _mm_load_ps(_Orinet), _mm_and_ps( _mm_castsi128_ps(_mm_cmplt_epi32(GyInt32,_m128i_Zero)) ,_m128_PI) );
            _mm_store_ps(_Orinet,Orientation);
            _Orinet+=4;
            _Gy+=8;
        }
        for(index=0;index<modA;++index)
        {
            *_Orinet += ((*_Gy) <0)*PI;
            ++_Orinet;
            ++_Gy;
        }
    }
    #ifdef DEBUG_SHOW

        // int Color[18] = {0xFFFFFF,0xC1FFC1,0xBFEFFF,0xB7B7B7,
        //                     0x9370DB,0x9ACD32,0x8E388E,0x5CACEE,
        //                     0xB8860B,0xFF6347,0xFF0000,0xFF00FF,
        //                     0xFFB90F,0x4B0082,0x4A708B,0x00CD00,
        //                     0x00008B,0x000000};

        float* OriData = Orient;

        for(int i=0;i<area;i+=1) 
        {
            int d = (int)(OriData[i]*255/2/PI);
            data[i] = (uchar)d;
        }
        imshow("OriG",DebugImg);
        cv::waitKey(0);
        #endif    
        #ifdef DEBUG_COUT
        cout<<" Orient "<<endl;
        for(int i=0;i<height;++i)
        {
            for(int j=0;j<width;++j)
            {
                cout<<Orient[i*width+j]<<" ";
            }
            cout<<endl;
        }
    #endif

    #ifdef DEBUG_FOUT
        ofstream ofs("./log.txt",ios::trunc);
        float _g , _o , _gx , _gy , cal_g , cal_o0 , cal_o1 , cal_o2;
        unsigned m=height*width;
        for(uint i=0;i<height;++i)
        {
            for(uint j=0;j<width;++j)
            {
                _gx=Gx[i*width+j]*0.5;
                _gy=Gy[i*width+j]*0.5;
                _g = Grad[i*width+j];
                cal_g= sqrt(_gx*_gx+_gy*_gy);
                _o = Orient[i*width+j];
                if(full)
                {
                    cal_o0 = _o/2/PI*360;
                    if(_gy<0)
                        cal_o1 = acos(_gx/(_g+1e-10f))+PI;
                    else
                        cal_o1 = acos(_gx/(_g+1e-10f));
                    cal_o2 = cal_o1/2/PI*360;
                }
                else
                {
                    cal_o0 = _o/PI*180;
                    cal_o1 = acos(_gx/(_g+1e-10f));
                    cal_o2 = cal_o1/2/PI*360;
                }
                ofs<<"x = "<<j<<" y = "<<i<<endl;
                ofs<<" gx "<<_gx<<" gy "<<_gy<<" _g "<<_g<<" cal g "<<cal_g<<endl;
                ofs<<" _o "<<_o<<" cal_o0 "<<cal_o0<<" cal_o1 "<<cal_o1<<" cal_o2 "<<cal_o2<<endl;
            }
        }
        ofs.close();
    #endif

    delete[] Gy;
    delete[] Gx;
}

void ConvAngel2Bin(float* Orient,float* Grad,int32_t* Orient0,int32_t* Orient1,
                    float* Grad0,float* Grad1,int h,int w,int cellSize,int nOrients,
                    bool full,bool interpolate)
{
    unsigned int i=0;
    int _int_o0,_int_o1;
    float _float_o, _float_od, _float_m;
    float _float_norm = 1.0f/cellSize/cellSize;
    __m128i _128i_o0, _128i_o1, *_128ip_O0, *_128ip_O1; 
    __m128 _128_o, _128_od, _128_m, *_128p_M0, *_128p_M1, *_128p_Orient, *_128p_Grad;
    const float oMult=(float)nOrients/(full?2*PI:PI); 
    const __m128 _m128_oMult=_mm_set1_ps(oMult);
    const __m128i _m128i_oMax=_mm_set1_epi32(nOrients);
    const __m128i _m128i_One=_mm_set1_epi32(1);
    const __m128i _m128i_Zero=_mm_set1_epi32(0);
    const __m128 _m128_Zero=_mm_set1_ps(0.f);
    const __m128 half=_mm_set1_ps(0.5f);
    const __m128 _m128_norm=_mm_set1_ps(_float_norm);

    const unsigned int aliq =  (h*w)/4;
    const unsigned int mod = (h*w)%4;
    _128p_Orient = (__m128*) Orient; _128p_Grad =  (__m128*) Grad;
    _128ip_O0=(__m128i*) Orient0; _128ip_O1=(__m128i*) Orient1; 
    _128p_M0=(__m128*) Grad0; _128p_M1=(__m128*) Grad1;
    if(interpolate) 
    {   
        for( i=0;i<aliq;i+=1) 
        {
            //原本的角度(0~2pi) × nOrients / (2×PI) // 角度归一化到 [0 ,nOrients]
            _128_o=_mm_mul_ps(*_128p_Orient,_m128_oMult); 
            // 取整数部分
            _128i_o0=_mm_cvttps_epi32(_128_o); 
            // 小数部分
            _128_od=_mm_sub_ps(_128_o,_mm_cvtepi32_ps(_128i_o0));
            // 以下一句可以不要？ 整数部分 < nOrients 返回_o0 否则返回 0
            // _128i_o0=_mm_and_si128(_mm_cmplt_epi32(_128i_o1,_m128i_oMax),_128i_o0); 
            // // 给O0赋值 
            *_128ip_O0++=_128i_o0;
            // 整数部分+1
            _128i_o1=_mm_add_epi32(_128i_o0,_m128i_One); 
            // (整数部分+1) < nOrients 
            _128i_o1=_mm_and_si128(_mm_cmplt_epi32(_128i_o1,_m128i_oMax),_128i_o1); 
            // 给O1赋值 
            *_128ip_O1++=_128i_o1;
            // 幅度除以Cell面积
            _128_m=_mm_mul_ps(*_128p_Grad,_m128_norm); 
            *_128p_M1=_mm_mul_ps(_128_od,_128_m); 
            *_128p_M0++=_mm_sub_ps(_128_m,*_128p_M1); 
            ++_128p_M1;

            ++_128p_Orient;
            ++_128p_Grad;
        }
        float* _floatp_Grad = (float*)_128p_Grad;
        float* _floatp_Orient = (float*)_128p_Orient;
        float* _floatp_M0 = (float*)_128p_M0;
        float* _floatp_M1 = (float*)_128p_M1;
        int32_t* _intp_o0 = (int32_t*)_128ip_O0;
        int32_t* _intp_o1 = (int32_t*)_128ip_O1;
        for(i=0;i<mod;++i)
        {
            _float_o=_floatp_Orient[i]*oMult; 
            _int_o0=(int) _float_o; 
            _float_od=_float_o-_int_o0;
            _intp_o0[i]=_int_o0;
            _int_o1=_int_o0+1; 
            if(_int_o1==nOrients) 
                _int_o1=0; 
            _intp_o1[i]=_int_o1;
            _float_m=_floatp_Grad[i]*_float_norm; 
            _floatp_M1[i]=_float_od*_float_m; 
            _floatp_M0[i]=_float_m-_floatp_M1[i];
        }

    }
    else
    {
        for( i=0;i<aliq;++i) 
        { 
            _128_o=_mm_mul_ps(*_128p_Orient,_m128_oMult); 
            _128i_o0=_mm_cvttps_epi32(_mm_add_ps(_128_o,half));
            _128i_o0=_mm_and_si128(_mm_cmplt_epi32(_128i_o0,_m128i_oMax),_128i_o0); 
            *_128ip_O0++=_128i_o0;
            _128_m=_mm_mul_ps(*_128p_Grad,_m128_norm); 
            *_128p_M0++=_128_m;             
            *_128p_M1++=_m128_Zero; 
            *_128ip_O1++=_m128i_Zero;

            ++_128p_Grad;
            ++_128p_Orient;
        }
        float* _floatp_Grad = (float*)_128p_Grad;
        float* _floatp_Orient = (float*)_128p_Orient;
        float* _floatp_M0 = (float*)_128p_M0;
        float* _floatp_M1 = (float*)_128p_M1;
        int32_t* _intp_o0 = (int32_t*)_128ip_O0;
        int32_t* _intp_o1 = (int32_t*)_128ip_O1;
        for(i=0;i<mod;++i)
        {
            _float_o = _floatp_Orient[i]*oMult; 
            _int_o0=(int) (_float_o+.5f);
            if(_int_o0>=nOrients) 
                _int_o0=0; 
            _intp_o0[i]=_int_o0;
            _floatp_M0[i]=_floatp_Grad[i]*_float_norm; 
            _floatp_M1[i]=0; 
            _intp_o1[i]=0;
        }
    }


    #ifdef DEBUG_FOUT
        ofstream ofs("./log2.txt",ios::trunc);
        for(int row=0;row<h;++row)
        {
            for(int col=0;col<w;++col)
            {
                ofs<<" x "<<col<<" y "<<row<<" O "<<Orient[row*w+col]<<" M "<<Grad[row*w+col]<<endl;
                ofs<<" O0 "<<Orient0[row*w+col]<<" O1 "<<Orient1[row*w+col]<<" M0 "<<Grad0[row*w+col]<<" M1 "<<Grad1[row*w+col]<<endl;
                int tem_o0 = Orient[row*w+col]/2/PI*nOrients;
                int tem_o1 = Orient[row*w+col]/2/PI*nOrients+1;
                ofs<<" C0 "<<tem_o0<<" C1 "<<tem_o1<<endl;
                if(tem_o1>=nOrients)
                    tem_o1 = 0;
                if(abs(tem_o0-Orient0[row*w+col])>1)
                {                
                    cout<<" x "<<col<<" y "<<row<<" O "<<Orient[row*w+col]<<" M "<<Grad[row*w+col]<<endl;
                    cout<<" O0 "<<Orient0[row*w+col]<<" O1 "<<Orient1[row*w+col]<<" M0 "<<Grad0[row*w+col]<<" M1 "<<Grad1[row*w+col]<<endl;
                    cout<<" C0 "<<tem_o0<<" C1 "<<tem_o1<<endl;
                }
            }
        }    
        ofs.close();
    #endif
}

void TrilinearInterpolation(float* Hog ,int32_t* Orient0,int32_t* Orient1,float* Grad0,float* Grad1,
                            unsigned int height,unsigned int width,unsigned int hogHeight,unsigned int hogWidth,
                            int cellSize,int nOrients)
{
    unsigned int hogStride = hogWidth * hogHeight ;
    // float* __attribute__((aligned(16))) Hog = new float[hogWidth*hogHeight*cellSize];
    // const int hb=height/bin, wb=width/bin, h0=hb*bin, w0=wb*bin, nb=wb*hb;
    float hy;
    int biny;
    float wy1,wy2;
    __m128 _m128_hx;
    float hx;
    float* __attribute__((aligned(16)))  wx1 = new float[4];
    float* __attribute__((aligned(16)))  wx2 = new float[4];
    int32_t*  __attribute__((aligned(16))) binx = new int32_t[4];
    __m128*  _m128p_wx1 = (__m128*)wx1;
    __m128*  _m128p_wx2 = (__m128*)wx2;
    __m128i* _m128ip_binx = (__m128i*)binx;
    int32_t *O0,*O1;
    float *M0,*M1;
    __m128 half = _mm_set1_ps(0.5f);
    __m128  _128_1_bin= _mm_set1_ps(1.0f/cellSize);
    __m128 _m128_One = _mm_set1_ps(1.0f);
    unsigned int aliqW = width/4;
    aliqW *= 4;
    unsigned int row,col;
    #define at(x,y,k) (Hog[(x) + (y) * hogWidth + (k) * hogStride])
    for(row=0; row<height; row++) 
    {
        hy = (row+0.5f)/cellSize-0.5f;
        biny = floor(hy);
        wy2 = hy - biny ;
        wy1 = 1.0 - wy2 ;
        O0 = Orient0+row*width;
        O1 = Orient1+row*width;
        M0 = Grad0+row*width;
        M1 = Grad1+row*width;
        // cout<<" hy "<<hy<<" biny "<<biny<<endl;
        // cout<<" debug BBB"<<endl;
        for(col=0; col<aliqW; col+=4)
        {
            // cout<<" col "<<col<<" "<<col+1<<" "<<col+2<<" "<<col+3<<endl;
            _m128_hx = _mm_set_ps(float (col), float (col+1), float (col+2), float (col+3));
            _m128_hx = _mm_sub_ps(_mm_mul_ps(_mm_add_ps(_m128_hx,half),_128_1_bin),half);
            *_m128ip_binx = _mm_cvttps_epi32(_m128_hx);
            // cout<<" binx "<<binx[0]<<" "<<binx[1]<<" "<<binx[2]<<" "<<binx[3]<<endl;
            *_m128p_wx2 = _mm_sub_ps(_m128_hx,_mm_cvtepi32_ps(*_m128ip_binx));
            *_m128p_wx1 = _mm_sub_ps(_m128_One,*_m128p_wx2);
            for(int i=0;i<4;++i)
            {               
                // cout<<" O0 "<<O0[i]<<endl; 
                if(binx[i] >= 0 && biny >=0)
                {
                    at(binx[i],biny,O0[i]) += M0[i] * wx1[i] * wy1 ;
                    at(binx[i],biny,O1[i]) += M1[i] * wx1[i] * wy1 ;
                }
                if(binx[i] < (signed)hogWidth - 1 && biny >=0)
                {
                    at(binx[i]+1,biny,O0[i]) += M0[i] * wx2[i] * wy1 ;
                    at(binx[i]+1,biny,O1[i]) += M1[i] * wx2[i] * wy1 ;
                }
                if (binx[i] < (signed)hogWidth - 1 && biny < (signed)hogHeight - 1) 
                {
                    at(binx[i]+1,biny+1,O0[i]) += M0[i] * wx2[i] * wy2 ;
                    at(binx[i]+1,biny+1,O1[i]) += M1[i] * wx2[i] * wy2 ;
                }
                if (binx[i] >= 0 && biny < (signed)hogHeight - 1) 
                {
                    at(binx[i],biny+1,O0[i]) += M0[i] * wx1[i] * wy2 ;
                    at(binx[i],biny+1,O1[i]) += M1[i] * wx1[i] * wy2 ;
                }
            }
            M0+=4;   M1+=4;     
            O0+=4;   O1+=4;  
        }
        // cout<<" debug ddd"<<endl;
        for(; col<width; col++)
        {
            hx = (col + 0.5) / cellSize - 0.5 ;
            binx[0] = floor(hx) ;
            wx2[0] = hx - binx[0] ;
            wx1[0] = 1.0 - wx2[0] ;

            if(binx[0] >= 0 && biny >=0)
            {
                at(binx[0],biny,O0[0]) += M0[0] * wx1[0] * wy1 ;
                at(binx[0],biny,O1[0]) += M0[0] * wx1[0] * wy1 ;
                at(binx[0],biny,O0[0]) += M1[0] * wx1[0] * wy1 ;
                at(binx[0],biny,O1[0]) += M1[0] * wx1[0] * wy1 ;
            }
            if(binx[0] < (signed)hogWidth - 1 && biny >=0)
            {
                at(binx[0]+1,biny,O0[0]) += M0[0] * wx2[0] * wy1 ;
                at(binx[0]+1,biny,O1[0]) += M0[0] * wx2[0] * wy1 ;
                at(binx[0]+1,biny,O0[0]) += M1[0] * wx2[0] * wy1 ;
                at(binx[0]+1,biny,O1[0]) += M1[0] * wx2[0] * wy1 ;
            }
            if(binx[0] < (signed)hogWidth - 1 && biny < (signed)hogHeight - 1) 
            {
                at(binx[0]+1,biny+1,O0[0]) += M0[0] * wx2[0] * wy2 ;
                at(binx[0]+1,biny+1,O1[0]) += M0[0] * wx2[0] * wy2 ;
                at(binx[0]+1,biny+1,O0[0]) += M1[0] * wx2[0] * wy2 ;
                at(binx[0]+1,biny+1,O1[0]) += M1[0] * wx2[0] * wy2 ;
            }
            if (binx[0] >= 0 && biny < (signed)hogHeight - 1) 
            {
                at(binx[0],biny+1,O0[0]) += M0[0] * wx1[0] * wy2 ;
                at(binx[0],biny+1,O1[0]) += M0[0] * wx1[0] * wy2 ;
                at(binx[0],biny+1,O0[0]) += M1[0] * wx1[0] * wy2 ;
                at(binx[0],biny+1,O1[0]) += M1[0] * wx1[0] * wy2 ;
            }
            ++M0;   ++M1;     
            ++O0;   ++O1;                
        }
    }
    delete[] wx1;
    delete[] wx2;
    delete[] binx;
}

float* hogNormMatrix( float *H, int nOrients, int hb, int wb, int cellSize) 
{
     // 最好平方和用SSE 求倒数不使用
    const float eps = 1e-4f/4/cellSize/cellSize/cellSize/cellSize; 
    const unsigned int hw = hb*wb;  
    float*  __attribute__((aligned(16))) N = new float[hw];
    float* N1 = N;
    int index, o, col, row;


    #ifdef SSE_ACC
        /****计算平方和****************/
        unsigned int aliqA = hw/4;
        aliqA *=4;

        __m128 _m128_H;
        __m128* _m128p_H = (__m128*)H;
        __m128 _m128_N,_m128_N1;
        // 第 0 方向
        for(index=0;index<aliqA;index+=4)
        {
            _mm_store_ps(N1,_mm_mul_ps(*_m128p_H,*_m128p_H));
            N1+=4;
            ++_m128p_H;
        }
        for(;index<hw;++index)
        {
            *N1 = H[index]*H[index];
            ++N1;
        }

        // [1,nOrients) 方向      
        float *H_nOri;  
        for( o=1; o<nOrients; o++) 
        {
            N1 = N;
            H_nOri = H + o * hw;
            for(index=0;index<aliqA;index+=4)
            {
                _m128_N = _mm_load_ps(N1);
                _m128_H = _mm_loadu_ps(H_nOri);
                _mm_store_ps(N1,_mm_add_ps(_m128_N, _mm_mul_ps(_m128_H,_m128_H)));
                N1+=4;
                H_nOri+=4;
            }
            for(;index<hw;++index)
            {
                *N1 = *H_nOri* *H_nOri;
                ++N1;
            }
        }

        /*****************计算 倒数**********************/
        float* prow, *prow1 , *rst;
        __m128 _m128_eps = _mm_set1_ps(eps);
        float* __attribute__((aligned(16))) N_cp = new float[hw];
        for(row=0;row<hb-1;++row)
        {   
            rst = N_cp+row*wb;
            prow = N+row*wb;
            prow1 = N+(row+1)*wb;
            for(col=0;col+4<=wb;col+=3)
            {
                _m128_N = _mm_loadu_ps(prow);
                _m128_N = _mm_add_ps(_m128_N,_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(_m128_N),4)));

                _m128_N1 = _mm_loadu_ps(prow1);
                _m128_N1 = _mm_add_ps(_m128_N1,_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(_m128_N1),4)));
                
                // _mm_storeu_ps(prow,_mm_add_ps(_m128_eps, _mm_add_ps(_m128_N,_m128_N1))));
                _mm_storeu_ps(rst, _mm_rsqrt_ps(_mm_add_ps(_m128_eps, _mm_add_ps(_m128_N,_m128_N1))));
                prow+=3;
                prow1+=3;
                rst+=3;
            }
            for(;col<wb-1;++col)
            {
                *rst = 1/ float(sqrt(*prow + *(prow+1) + *prow1 + *(prow1+1) +eps));
                ++prow;
                ++prow1;
                ++rst;
            }
        }
        // 最后一行 最后一列处理
        prow = N_cp+ (hb-1)*wb;
        prow1 = N_cp+ (hb-2)*wb;
        memcpy(prow,prow1,sizeof(float)*(wb-1));

        for(row=0;row<hb-1;++row)
        {
            prow =  N_cp+(row+1)*wb-1;
            prow1 =  N_cp+(row+1)*wb-2;
            *prow = *prow1;
        }
        prow = N_cp+hw-1;
        prow1 = N_cp+hw -wb -2;
        *prow = *prow1;
        delete[] N;
        return N_cp;
    #else
        float  *n , *n1; 
        for( o=0; o<nOrients; o++ ) 
        {
            int owh = o*hw;
            for( index=0; index<hw; index++ ) 
                    N[index] += H[owh+index]*H[owh+index];
        }
        // 四个方格求和再根号
        for( row=0; row<hb-1; row++ ) 
            for( col=0; col<wb-1; col++ ) 
            {
                n=N+row*wb+col; 
                *n=1/float(sqrt(n[0]+n[1]+n[hb]+n[hb+1]+eps)); 
            }
        n = N+ (hb-1)*wb;
        n1 = N+ (hb-2)*wb;
        memcpy(n,n1,sizeof(float)*(wb-1));
    
        for(row=0;row<hb-1;++row)
        {
            n =  N+(row+1)*wb-1;
            n1 =  N+(row+1)*wb-2;
            *n = *n1;
        }
        n = N+hw-1;
        n1 = N+hw -wb -2;
        *n = *n1;
        return N;
    #endif
}

// HOG helper: compute HOG or FHOG channels
void hogChannels( float *H, float *R, float *N,
    int hb, int wb, int nOrients, float clip, int type)
{
    const float r=.2357f; 
    int o, index,row,col; 
    float t;
    const int nb=wb*hb, nbo=nOrients*nb;
    int aliqA = nb/4;
    aliqA*=4;
    float *H1, *R1, *N1;
    #ifdef SSE_ACC
    {
        int aliqA1 = wb*(hb-1)/4;
        aliqA1= aliqA1*4 + wb;
        __m128 _m128_Mul,_m128_Cmp;
        __m128 _m128_clip = _mm_set1_ps(clip);
        __m128 half = _mm_set1_ps(0.5f);
        if( type==0) 
        {
            for( o=0; o<nOrients; o++ ) 
            {
                {   // 0
                    H1 = H+o*nb;
                    R1 = R+o*nb;
                    N1 = N;
                    for(index =0;index<aliqA;index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_load_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)));
                        H1+=4;
                        R1+=4;
                        N1+=4;
                    }
                    for(;index<nb;++index)
                    {
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 = t;
                        ++H1;   ++R1;   ++N1;
                    }
                }
                {   // 1
                    // 第一行特殊处理
                    H1 = H+nbo+o*nb;
                    R1 = R+o*nb;
                    N1 = N;
                    for(index=0; index+4<=wb; index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_load_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)));
                        H1+=4;
                        R1+=4;
                        N1+=4;                        
                    }
                    for(; index<wb; ++index)
                    {
                        
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 = t;
                        ++H1;   ++R1;   ++N1;                      
                    }
                    // 其余行
                    N1 = N;
                    for(; index<aliqA1; index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_load_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)));
                        H1+=4;
                        R1+=4;
                        N1+=4;                        
                    }                    
                    for(; index<nb; ++index)
                    {
                        
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 = t;
                        ++H1;   ++R1;   ++N1;                      
                    }

                }
                {   // 2
                    H1 = H+nb*o+2*nbo;
                    float *R1 = R+o*nb;
                    float *N1 = N;
                    for( row=0; row<hb; ++row ) 
                    {
                        // 第一列特殊处理
                        t = R1[0]*N1[0];
                        *H1 =  (t>clip) ? clip:t;
                        // 其他列
                        ++H1;
                        ++R1;
                        for( col=1; col+4<=wb; col+=4 ) 
                        {
                            _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                            _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                            _mm_storeu_ps(H1, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)));
                            H1+=4;
                            R1+=4;
                            N1+=4;
                        }
                        for( ; col<wb; ++col ) 
                        {
                            t = *R1 * *N1;
                            if(t>clip)
                                t=clip;
                            *H1 = t;
                            ++H1;   ++R1;   ++N1;
                        } 
                        ++N1;
                    }
                }
                {   // 3
                    // 第一行 第一列
                    float *H1 = H+nb*o+3*nbo;
                    float *R1 = R+o*nb;
                    float *N1 = N;
                    t = *R1 * *N1;
                    if(t>clip)
                        t=clip;
                    *H1 = t;
                    //第一行其他列 
                    ++H1;
                    ++R1;
                    for(index=1; index+4<=wb; index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)));
                        H1+=4;
                        R1+=4;
                        N1+=4;                        
                    }
                    for(; index<wb; ++index)
                    {
                        
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 = t;
                        ++H1;   ++R1;   ++N1;                      
                    }     
                    N1=N;
                    for(row=1;row<hb;++row)
                    {
                        // 第一列
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 = t;
                        ++H1;
                        ++R1;
                        for(col=1;col+4<=wb;col+=4)
                        {
                            _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                            _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                            _mm_storeu_ps(H1, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)));
                            H1+=4;
                            R1+=4;
                            N1+=4;  
                        }
                        for(;col<wb;++col)
                        {
                            t = *R1 * *N1;
                            if(t>clip)
                                t=clip;
                            *H1 = t;
                            ++H1;
                            ++R1;
                            ++N1;  
                        }
                        ++N1;  
                    }
                }
            }
        }
    }
    #else 
    {
        N1 = N;
        for( o=0; o<nOrients; o++ ) 
        {
            R1 = R+ o*nb;
            { // 0
                H1 = H+ o*nb;
                for( index=0; index<nb; ++index ) 
                {
                    t = R1[index]*N1[index];
                    if(t>clip) 
                        t=clip;
                    H1[index] = t;
                }
            }
            { // 1
                H1 = H+o*nb+nbo;
                // 第一行处理
                for(index=0;index<wb;++index)
                { 
                    t = R1[index]*N1[index];
                    if(t>clip) 
                        t=clip;
                    H1[index] = t;
                }

                for(;index<nb;++index)
                { 
                    t = R1[index]*N1[index-wb];
                    if(t>clip) 
                        t=clip;
                    H1[index] = t;
                }
            }
            { // 2
                H1 = H+o*nb+2*nbo;
                cout<<"HJ"<<endl;
                for(row=0;row<hb;++row)
                {
                    // 第一列处理
                    int rw = row*wb;
                    t = R1[rw] * N1[rw];
                    if(t>clip) 
                        t=clip;
                    H1[rw] = t;
                    cout<<N1[rw]<<" ";
                    // 其他列处理
                    for(col=1;col<wb;++col)
                    {
                        t = R1[rw+col] * N1[rw+col-1];
                        cout<<N1[rw+col-1]<<" ";
                        if(t>clip) 
                            t=clip;
                        H1[rw+col] = t;
                    }
                    cout<<endl;
                }
            }
            { // 3
                H1 = H+o*nb+3*nbo;
                //第一行第一列
                t = R1[0]*N1[0];
                if(t>clip) 
                    t=clip;
                H1[0] = t;
                //第一行 其他
                for(col=1;col<wb;++col)
                {
                    t = R1[col] * N[col-1];
                    if(t>clip) 
                        t=clip;
                    H1[col] = t;
                }
                for(row=1;row<hb;++row)
                {
                    // 第一列处理
                    int rw = row*wb;
                    int r1w = (row-1)*wb;
                    t = R1[rw] * N1[r1w];
                    if(t>clip) 
                        t=clip;
                    H1[rw] = t;
                    // 其他列处理
                    for(col=1;col<wb;++col)
                    {
                        t = R1[rw+col] * N1[r1w+col-1];
                        if(t>clip) 
                            t=clip;
                        H1[rw+col] = t;
                    }
                }
            }
        }
    }
    #endif
}

int main()
{
    cv::Mat img(imread("/home/huajun/Desktop/hog.jpg",cv::IMREAD_GRAYSCALE));

    return 0;
}
