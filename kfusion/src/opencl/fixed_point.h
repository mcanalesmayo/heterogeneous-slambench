#define FRACT_BITS_J 8
#define FRACT_BITS_J_D2 FRACT_BITS_J/2
#define FRACT_BITS_ERROR 24
#define FRACT_BITS_ERROR_D2 FRACT_BITS_ERROR/2
#define FIXED_ONE ( 1 << FRACT_BITS )
#define INT2FIXED(x, q) ( (x) << q )
#define FLOAT2FIXED(x, q) ( (long)((x) * (1L << q)) )
#define FIXED2INT(x, q) ( (x) >> q )
#define FIXED2FLOAT(x, q) ( ((float)(x)) / (1L << q) )
#define PREPARE_FOR_MULT(x, q_d2) ( (x >> q_d2) )
#define PREPARED_MULT(x_d2, y_d2) ( x_d2 * y_d2 )
#define MULT(x, y, q) ( ((x >> (q/2)) * (y >> (q/2))) )

typedef struct sTrackDataFixedPoint {
	int result;
	int error;
	int J[6];
} TrackDataFixedPoint;