#include <emmintrin.h>
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <fstream>

using namespace std;

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

void CalGrad(cv::Mat& img , float* Grad ,float* Orient )
{
    assert(img.channels() == 1);
    const long unsigned int width = img.cols,height = img.rows;
    assert(width*height < 4.29e9);

    const long unsigned int aliquot = (width * height)/16;  
    const long unsigned int mod = (width * height)%16;
    __m128i ucharData,lower8Byte,upper8Byte;
    __m128i zeros = _mm_setzero_si128();

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
        __m128 GxFloat,GyFloat,Gradient,Orientation;
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

            float debug[4];
            _mm_storeu_ps(debug,GxFloat);
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

            _mm_storeu_ps(debug,GxFloat);
            
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
    // // 输出 梯度幅度
    // for(int i=0;i<height;++i)
    // {
    //     for(int j=0;j<width;++j)
    //     {
    //         cout<<Grad[i*width+j]<<" ";
    //     }
    //     cout<<endl;
    // }   
    // cout<<endl;

    // for(int i=0;i<height;++i)
    // {
    //     for(int j=0;j<width;++j)
    //     {
    //         cout<<Orient[i*width+j]<<" ";
    //     }
    //     cout<<endl;
    // }   
    // cout<<endl;

    delete[] Gy;
    delete[] Gx;
}



int main()
{
    cv::Mat img(imread("/home/huajun/Desktop/track_map.png",cv::IMREAD_GRAYSCALE));

    // cout<<img<<endl;
    const long unsigned int width = img.cols;
    const long unsigned int height = img.rows;

    // cout<<img<<endl;
    double start = (double)cv::getTickCount();

    float* __attribute__((aligned(16))) Grad = new float[width*height];
    float* __attribute__((aligned(16))) Orient = new float[width*height];

    CalGrad(img,Grad,Orient);
    double cost = ((double)cv::getTickCount()-start)/(double)cv::getTickFrequency();
    cout<<"cost "<<cost*10e2<<" ms "<<endl;


    cv::Mat out(img.rows,img.cols,CV_32FC1,Grad);

    // imwrite("/home/huajun/Desktop/hhhh.jpg",img);
    // imshow("HJ",out);
    // cv::waitKey(0);    

    delete[] Grad;
    delete[] Orient;
    return 0;
}