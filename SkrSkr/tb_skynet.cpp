#include "skynet_flow.h"

template<
	unsigned N_BATCH,
	unsigned VEC_LEN
>
void load_ifm(data_stream<BIT_ACTV>& ifm)
{
	uint8_t* ifm_buffer;
	ifm_buffer =  (uint8_t*)malloc(VEC_LEN*sizeof(uint8_t));
	#ifdef HLS_CSIM
	FILE *fp = fopen("../../../../data/Raw_u8.bin", "rb");
	#else
	FILE *fp = fopen("../data/Raw_u8.bin", "rb");
	#endif
	fread(ifm_buffer, 1, VEC_LEN*sizeof(uint8_t), fp);
	fclose(fp);

	for(size_t b=0; b < N_BATCH; b++)
	{
		for(size_t i=0; i < VEC_LEN; i++)
		{
			ifm.write(ifm_buffer[i]);
		}
	}

	free(ifm_buffer);
}

template<
	unsigned N_BATCH,
	unsigned VEC_LEN
>
int32_t check_ofm(axiu_stream<BIT_CONV>& ofm)
{
	int16_t* ofm_buffer;
	ofm_buffer =  (int16_t*)malloc(VEC_LEN*sizeof(int16_t));
	#ifdef HLS_CSIM
	FILE *fp = fopen("../../../../data/Out_s16.bin", "rb");
	#else
	FILE *fp = fopen("../data/Out_s16.bin", "rb");
	#endif
	fread(ofm_buffer, 1, VEC_LEN*sizeof(int16_t), fp);
	fclose(fp);

	int16_t temp;
	int32_t error_cnt = 0;
	for(size_t b=0; b < N_BATCH; b++)
	{
		for(size_t i=0; i < VEC_LEN; i++)
		{
			temp = (int16_t)ofm.read().data;
			//std::cout << i << " " << (int16_t)temp << " " << ofm_buffer[i] << std::endl;
			if(temp!=ofm_buffer[i])
			{
				//std::cout << i << " " << (int16_t)temp << " " << ofm_buffer[i] << std::endl;
				error_cnt++;
			}
		}
	}

	free(ofm_buffer);
	return error_cnt;
}

int main()
{
	data_stream<BIT_ACTV> s_in_1;
	data_stream<4 * BIT_ACTV> s_in_4;
	load_ifm<
		N_BATCH,
		ROWIN * COLIN * L0_DW_NCH
	>(s_in_1);
	printf("in size: %lu\n", s_in_1.size());

	expandWidth<
		BIT_ACTV,
		N_IN * BIT_ACTV, 
		ROWIN * COLIN * L0_DW_NCH * N_BATCH
	>(s_in_1, s_in_4);

	axiu_stream<BIT_CONV> s_l0_dwconv_out("s_l0_dwconv_out");
	skynet_flow(s_in_4, s_l0_dwconv_out);
	
	int32_t error_cnt = check_ofm< N_BATCH, 14 >(s_l0_dwconv_out);
	
	if(error_cnt > 0)
	{
		std::cout << "Test Failed! " << error_cnt << " errors" << std::endl;
		return 1;
	}
	else
	{
		std::cout << "Test Passed! " << std::endl;
		return 0;
	}
}
