#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include "svm.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int libsvm_version = LIBSVM_VERSION;
typedef float Qfloat;
typedef signed char schar;
#define min(x, y) ( ((x) < (y)) ? (x) : (y) )
#define max(x, y) ( ((x) > (y)) ? (x) : (y) )
#define swap(x, y) do { typeof(x) SWAP = x; x = y; y = SWAP; } while (0)
#define clone(dst, src, n) do \
{ dst = malloc(sizeof(typeof(*dst)) * n); memcpy((void *)dst,(void *)src,sizeof(typeof(*dst))*n); } while (0)

static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

typedef struct svm_node svm_node;
typedef struct svm_problem svm_problem;
typedef struct svm_parameter svm_parameter;
typedef struct svm_model svm_model;

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
typedef struct head_t_Struct
{
    struct head_t_Struct *prev, *next;	// a circular list
    Qfloat *data;
    int len;		// data[0,len) is cached in this entry
} head_t;
typedef struct Cache_Struct
{
//public:
//	Cache(int l,long int size);
//	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
//	int get_data(const int index, Qfloat **data, int len);
//	void swap_index(int i, int j);
//private:
	int l;
	long int size;


	head_t *head;
	head_t lru_head;
//	void lru_delete(head_t *h);
//	void lru_insert(head_t *h);
} Cache;

Cache* Cache__new(int l,long int size) {
    Cache* this = malloc(sizeof(Cache));
    this->head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
    this->size /= sizeof(Qfloat);
    this->size -= l * sizeof(head_t) / sizeof(Qfloat);
    this->size = max(size, 2 * (long int) l);	// cache must be large enough for two columns
    this->lru_head.next = this->lru_head.prev = &this->lru_head;
    return this;
}

void Cache__destructor(Cache* this)
{
	for(head_t *h = this->lru_head.next; h != &this->lru_head; h=h->next)
		free(h->data);
	free(this->head);
}

void Cache__lru_delete(Cache* this, head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache__lru_insert(Cache* this, head_t *h)
{
	// insert to last position
	h->next = &this->lru_head;
	h->prev = this->lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int Cache__get_data(Cache* this, const int index, Qfloat **data, int len)
{
	head_t *h = &this->head[index];
	if(h->len) Cache__lru_delete(this, h);
	int more = len - h->len;

	if(more > 0)
	{
		// free old space
		while(this->size < more)
		{
			head_t *old = this->lru_head.next;
			Cache__lru_delete(this, old);
			free(old->data);
			this->size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		this->size -= more;
		swap(h->len,len);
	}

	Cache__lru_insert(this, h);
	*data = h->data;
	return len;
}

void Cache__swap_index(Cache* this, int i, int j)
{
	if(i==j) return;

	if(this->head[i].len) Cache__lru_delete(this, &this->head[i]);
	if(this->head[j].len) Cache__lru_delete(this, &this->head[j]);
	swap(this->head[i].data,this->head[j].data);
	swap(this->head[i].len,this->head[j].len);
	if(this->head[i].len) Cache__lru_insert(this, &this->head[i]);
	if(this->head[j].len) Cache__lru_insert(this, &this->head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = this->lru_head.next; h!=&this->lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else
			{
				// give up
                Cache__lru_delete(this, h);
				free(h->data);
                this->size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
typedef struct QMatrix_Struct QMatrix;
typedef struct SVC_Q_extends_Kernel_Struct SVC_Q;
typedef struct ONE_CLASS_Q_extends_Kernel_Struct ONE_CLASS_Q;
typedef struct SVR_Q_extends_Kernel_Struct SVR_Q;

struct QMatrix_Struct {
/*
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};
*/
};
Qfloat *QMatrix_virtual__get_Q(SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, int column, int len);
double *QMatrix_virtual__get_QD(SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q);
void QMatrix_virtual__swap_index(SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, int i, int j);

Qfloat *SVC_Q__get_Q(SVC_Q* this, int i, int len);
Qfloat *ONE_CLASS_Q__get_Q(ONE_CLASS_Q* this, int i, int len);
Qfloat *SVR_Q__get_Q(SVR_Q* this, int i, int len);
Qfloat *QMatrix_virtual__get_Q(SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, int column, int len) {
    if(svc_q) {
        return SVC_Q__get_Q(svc_q, column, len);
    }
    if(one_class_q) {
        return ONE_CLASS_Q__get_Q(one_class_q, column, len);
    }
    if(svr_q) {
        return SVR_Q__get_Q(svr_q, column, len);
    }
    exit(11);
}

double *SVC_Q__get_QD(SVC_Q* this);
double *ONE_CLASS_Q__get_QD(ONE_CLASS_Q* this);
double *SVR_Q__get_QD(SVR_Q* this);
double *QMatrix_virtual__get_QD(SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q) {
    if(svc_q) {
        return SVC_Q__get_QD(svc_q);
    }
    if(one_class_q) {
        return ONE_CLASS_Q__get_QD(one_class_q);
    }
    if(svr_q) {
        return SVR_Q__get_QD(svr_q);
    }
    exit(12);
}

void SVC_Q__swap_index(SVC_Q* this, int i, int j);
void ONE_CLASS_Q__swap_index(ONE_CLASS_Q* this, int i, int j);
void SVR_Q__swap_index(SVR_Q* this, int i, int j);
void QMatrix_virtual__swap_index(SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, int i, int j) {
    if(svc_q) {
        return SVC_Q__swap_index(svc_q, i, j);
    }
    if(one_class_q) {
        return ONE_CLASS_Q__swap_index(one_class_q, i, j);
    }
    if(svr_q) {
        return SVR_Q__swap_index(svr_q, i, j);
    }
    exit(12);
}


typedef struct Kernel_extends_QMatrix_Struct Kernel;
struct Kernel_extends_QMatrix_Struct {
    QMatrix *superQMatrix;
//public:
//    Kernel(int l, svm_node * const * x, const svm_parameter& param);
//    virtual ~Kernel();

//	static double k_function(const svm_node *x, const svm_node *y,
//				 const svm_parameter& param);
//	virtual Qfloat *get_Q(int column, int len) const = 0;
//	virtual double *get_QD() const = 0;
//	virtual void swap_index(int i, int j) const	// no so const...
//	{
//		swap(x[i],x[j]);
//		if(x_square) swap(x_square[i],x_square[j]);
//	}
//protected:

//	double (Kernel::*kernel_function)(int i, int j) const;
    double (*kernel_function)(Kernel *this,int i, int j);

//private:
	const struct svm_node **x;
	double *x_square;

	// svm_parameter
	int kernel_type;
	int degree;
	double gamma;
	double coef0;

};
void Kernel__swap_index(Kernel* this, int i, int j) {
    swap(this->x[i],this->x[j]);
    if(this->x_square) swap(this->x_square[i],this->x_square[j]);
}
static double Kernel__dot(const struct svm_node *px, const struct svm_node *py);
double Kernel__kernel_linear(Kernel* this, int i, int j)
{
    return Kernel__dot(this->x[i],this->x[j]);
}
double Kernel__kernel_poly(Kernel* this, int i, int j)
{
    return powi(this->gamma*Kernel__dot(this->x[i],this->x[j])+this->coef0,this->degree);
}
double Kernel__kernel_rbf(Kernel* this, int i, int j)
{
    return exp(-this->gamma*(this->x_square[i]+this->x_square[j]-2*Kernel__dot(this->x[i],this->x[j])));
}
double Kernel__kernel_sigmoid(Kernel* this, int i, int j)
{
    return tanh(this->gamma*Kernel__dot(this->x[i],this->x[j])+this->coef0);
}
double Kernel__kernel_precomputed(Kernel* this, int i, int j)
{
    return this->x[i][(int)(this->x[j][0].value)].value;
}

Kernel* Kernel__new(int l, struct svm_node * const * x_, const struct svm_parameter param)
//:kernel_type(param.kernel_type), degree(param.degree),
// gamma(param.gamma), coef0(param.coef0)
{
    Kernel* this = malloc(sizeof(Kernel));this->superQMatrix = Malloc(QMatrix, 1);
    this->kernel_type = param.kernel_type;
    this->degree = param.degree;
    this->gamma = param.gamma;
    this->coef0 = param.coef0;
	switch(this->kernel_type)
	{
		case LINEAR:
			this->kernel_function = &Kernel__kernel_linear;
			break;
		case POLY:
            this->kernel_function = &Kernel__kernel_poly;
			break;
		case RBF:
            this->kernel_function = &Kernel__kernel_rbf;
			break;
		case SIGMOID:
            this->kernel_function = &Kernel__kernel_sigmoid;
			break;
		case PRECOMPUTED:
            this->kernel_function = &Kernel__kernel_precomputed;
			break;
	}

	clone(this->x,x_,l);

	if(this->kernel_type == RBF)
	{
        this->x_square = malloc(sizeof(double) * l);
		for(int i=0;i<l;i++)
            this->x_square[i] = Kernel__dot(this->x[i],this->x[i]);
	}
	else
        this->x_square = 0;

    return this;
}

void Kernel__destructor(Kernel* this)
{
	free(this->x);
	free(this->x_square);
}

double Kernel__dot(const struct svm_node *px, const struct svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}
	}
	return sum;
}

double Kernel__k_function(const struct svm_node *x, const struct svm_node *y,
			  const struct svm_parameter param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return Kernel__dot(x,y);
		case POLY:
			return powi(param.gamma*Kernel__dot(x,y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}

			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*Kernel__dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable
	}
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
typedef struct SolutionInfo_Struct SolutionInfo;
struct SolutionInfo_Struct {
    double obj;
    double rho;
    double upper_bound_p;
    double upper_bound_n;
    double r;	// for Solver_NU
};

typedef struct Solver_Struct Solver;
typedef struct Solver_NU_extends_Solver_Struct Solver_NU;
struct Solver_Struct {
//public:
//	Solver() {};
//	virtual ~Solver() {};



//	void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
//		   double *alpha_, double Cp, double Cn, double eps,
//		   SolutionInfo* si, int shrinking);
//protected:
	int active_size;
	schar *y;
	double *G;		// gradient of objective function
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;
//	const QMatrix *Q;
	const double *QD;
	double eps;
	double Cp,Cn;
	double *p;
	int *active_set;
	double *G_bar;		// gradient, if we treat free variables as 0
	int l;
	bool unshrink;	// XXX

//	double get_C(int i)
//	{
//		return (y[i] > 0)? Cp : Cn;
//	}
//	void update_alpha_status(int i)
//	{
//		if(alpha[i] >= get_C(i))
//			alpha_status[i] = UPPER_BOUND;
//		else if(alpha[i] <= 0)
//			alpha_status[i] = LOWER_BOUND;
//		else alpha_status[i] = FREE;
//	}
//	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
//	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
//	bool is_free(int i) { return alpha_status[i] == FREE; }
//	void swap_index(int i, int j);
//	void reconstruct_gradient();
//	virtual int select_working_set(int &i, int &j);
//	virtual double calculate_rho();
//	virtual void do_shrinking();
//private:
//	bool be_shrunk(int i, double Gmax1, double Gmax2);
};
int Solver_virtual__select_working_set(Solver *solver, Solver_NU *solver_nu, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, int *out_i, int *out_j);
double Solver_virtual__calculate_rho(Solver *solver, Solver_NU *solver_nu);
void Solver_virtual__do_shrinking(Solver *solver, Solver_NU *solver_nu, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q);

double Solver__get_C(Solver* this, int i)
{
    return (this->y[i] > 0)? this->Cp : this->Cn;
}

void Solver__update_alpha_status(Solver* this, int i)
{
    if(this->alpha[i] >= Solver__get_C(this, i))
        this->alpha_status[i] = UPPER_BOUND;
    else if(this->alpha[i] <= 0)
        this->alpha_status[i] = LOWER_BOUND;
    else this->alpha_status[i] = FREE;
}

bool Solver__is_upper_bound(Solver* this, int i) { return this->alpha_status[i] == UPPER_BOUND; }

bool Solver__is_lower_bound(Solver* this, int i) { return this->alpha_status[i] == LOWER_BOUND; }

bool Solver__is_free(Solver* this, int i) { return this->alpha_status[i] == FREE; }

void Solver__swap_index(Solver* this, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, int i, int j)
{
    QMatrix_virtual__swap_index(svc_q, one_class_q, svr_q, i,j);
	swap(this->y[i],this->y[j]);
	swap(this->G[i],this->G[j]);
	swap(this->alpha_status[i],this->alpha_status[j]);
	swap(this->alpha[i],this->alpha[j]);
	swap(this->p[i],this->p[j]);
	swap(this->active_set[i],this->active_set[j]);
	swap(this->G_bar[i],this->G_bar[j]);
}

void Solver__reconstruct_gradient(Solver* this, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q)
{
	// reconstruct inactive elements of G from G_bar and free variables

	if(this->active_size == this->l) return;

	int i,j;
	int nr_free = 0;

	for(j=this->active_size;j<this->l;j++)
        this->G[j] = this->G_bar[j] + this->p[j];

	for(j=0;j<this->active_size;j++)
		if(Solver__is_free(this, j))
			nr_free++;

	if(2*nr_free < this->active_size)
		info("\nWARNING: using -h 0 may be faster\n");

	if (nr_free*this->l > 2*this->active_size*(this->l-this->active_size))
	{
		for(i=this->active_size;i<this->l;i++)
		{
			const Qfloat *Q_i = QMatrix_virtual__get_Q(svc_q, one_class_q, svr_q, i,this->active_size);
			for(j=0;j<this->active_size;j++)
				if(Solver__is_free(this, j))
                    this->G[i] += this->alpha[j] * Q_i[j];
		}
	}
	else
	{
		for(i=0;i<this->active_size;i++)
			if(Solver__is_free(this, i))
			{
				const Qfloat *Q_i = QMatrix_virtual__get_Q(svc_q, one_class_q, svr_q, i,this->l);
				double alpha_i = this->alpha[i];
				for(j=this->active_size;j<this->l;j++)
                    this->G[j] += alpha_i * Q_i[j];
			}
	}
}

void Solver__Solve(Solver* this, Solver_NU *this_solver_nu, int l, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
{
	this->l = l;
//	this->Q = Q;
	this->QD= QMatrix_virtual__get_QD(svc_q, one_class_q, svr_q);
	clone(this->p, p_,l);
	clone(this->y, y_,l);
	clone(this->alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	this->unshrink = false;

	// initialize alpha_status
	{
        this->alpha_status = malloc(sizeof(char) * l);
		for(int i=0;i<l;i++)
            Solver__update_alpha_status(this, i);
	}

	// initialize active set (for shrinking)
	{
        this->active_set = malloc(sizeof(int) * l);
		for(int i=0;i<l;i++)
            this->active_set[i] = i;
        this->active_size = l;
	}

	// initialize gradient
	{
        this->G = malloc(sizeof(double) * l);
        this->G_bar = malloc(sizeof(double) * l);
		int i;
		for(i=0;i<l;i++)
		{
            this->G[i] = this->p[i];
            this->G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
			if(!Solver__is_lower_bound(this, i))
			{
				const Qfloat *Q_i = QMatrix_virtual__get_Q(svc_q, one_class_q, svr_q, i,l);
				double alpha_i = this->alpha[i];
				int j;
				for(j=0;j<l;j++)
                    this->G[j] += alpha_i*Q_i[j];
				if(Solver__is_upper_bound(this, i))
					for(j=0;j<l;j++)
                        this->G_bar[j] += Solver__get_C(this, i) * Q_i[j];
			}
	}

	// optimization step

	int iter = 0;
	int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
	int counter = min(l,1000)+1;

	while(iter < max_iter)
	{
		// show progress and do shrinking

		if(--counter == 0)
		{
			counter = min(l,1000);
			if(shrinking) Solver_virtual__do_shrinking(this, this_solver_nu, svc_q, one_class_q, svr_q);
			info(".");
		}

		int i,j;
		if(Solver_virtual__select_working_set(this, this_solver_nu, svc_q, one_class_q, svr_q, &i, &j)!=0)
		{
			// reconstruct the whole gradient
            Solver__reconstruct_gradient(this, svc_q, one_class_q, svr_q);
			// reset active set size and check
			this->active_size = l;
			info("*");
			if(Solver_virtual__select_working_set(this, this_solver_nu, svc_q, one_class_q, svr_q, &i, &j)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}

		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully

        const Qfloat *Q_i = QMatrix_virtual__get_Q(svc_q, one_class_q, svr_q, i, this->active_size);
		const Qfloat *Q_j = QMatrix_virtual__get_Q(svc_q, one_class_q, svr_q,j,this->active_size);

		double C_i = Solver__get_C(this, i);
		double C_j = Solver__get_C(this, j);

		double old_alpha_i = this->alpha[i];
		double old_alpha_j = this->alpha[j];

		if(this->y[i]!=this->y[j])
		{
			double quad_coef = this->QD[i]+this->QD[j]+2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (-this->G[i]-this->G[j])/quad_coef;
			double diff = this->alpha[i] - this->alpha[j];
            this->alpha[i] += delta;
            this->alpha[j] += delta;

			if(diff > 0)
			{
				if(this->alpha[j] < 0)
				{
                    this->alpha[j] = 0;
                    this->alpha[i] = diff;
				}
			}
			else
			{
				if(this->alpha[i] < 0)
				{
                    this->alpha[i] = 0;
                    this->alpha[j] = -diff;
				}
			}
			if(diff > C_i - C_j)
			{
				if(this->alpha[i] > C_i)
				{
                    this->alpha[i] = C_i;
                    this->alpha[j] = C_i - diff;
				}
			}
			else
			{
				if(this->alpha[j] > C_j)
				{
                    this->alpha[j] = C_j;
                    this->alpha[i] = C_j + diff;
				}
			}
		}
		else
		{
			double quad_coef = this->QD[i]+this->QD[j]-2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (this->G[i]-this->G[j])/quad_coef;
			double sum = this->alpha[i] + this->alpha[j];
            this->alpha[i] -= delta;
            this->alpha[j] += delta;

			if(sum > C_i)
			{
				if(this->alpha[i] > C_i)
				{
                    this->alpha[i] = C_i;
                    this->alpha[j] = sum - C_i;
				}
			}
			else
			{
				if(this->alpha[j] < 0)
				{
                    this->alpha[j] = 0;
                    this->alpha[i] = sum;
				}
			}
			if(sum > C_j)
			{
				if(this->alpha[j] > C_j)
				{
                    this->alpha[j] = C_j;
                    this->alpha[i] = sum - C_j;
				}
			}
			else
			{
				if(this->alpha[i] < 0)
				{
                    this->alpha[i] = 0;
                    this->alpha[j] = sum;
				}
			}
		}

		// update G

		double delta_alpha_i = this->alpha[i] - old_alpha_i;
		double delta_alpha_j = this->alpha[j] - old_alpha_j;

		for(int k=0;k<this->active_size;k++)
		{
            this->G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}

		// update alpha_status and G_bar

		{
			bool ui = Solver__is_upper_bound(this, i);
			bool uj = Solver__is_upper_bound(this, j);
            Solver__update_alpha_status(this, i);
            Solver__update_alpha_status(this, j);
			int k;
			if(ui != Solver__is_upper_bound(this, i))
			{
				Q_i = QMatrix_virtual__get_Q(svc_q, one_class_q, svr_q, i,l);
				if(ui)
					for(k=0;k<l;k++)
                        this->G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
                        this->G_bar[k] += C_i * Q_i[k];
			}

			if(uj != Solver__is_upper_bound(this, j))
			{
				Q_j = QMatrix_virtual__get_Q(svc_q, one_class_q, svr_q, j,l);
				if(uj)
					for(k=0;k<l;k++)
                        this->G_bar[k] -= C_j * Q_j[k];
				else
					for(k=0;k<l;k++)
                        this->G_bar[k] += C_j * Q_j[k];
			}
		}
	}

	if(iter >= max_iter)
	{
		if(this->active_size < l)
		{
			// reconstruct the whole gradient to calculate objective value
            Solver__reconstruct_gradient(this, svc_q, one_class_q, svr_q);
            this->active_size = l;
			info("*");
		}
		fprintf(stderr,"\nWARNING: reaching max number of iterations\n");
	}

	// calculate rho

	si->rho = Solver_virtual__calculate_rho(this, this_solver_nu);

	// calculate objective value
	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += this->alpha[i] * (this->G[i] + this->p[i]);

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[this->active_set[i]] = this->alpha[i];
	}

	// juggle everything back
	/*{
		for(int i=0;i<l;i++)
			while(active_set[i] != i)
				swap_index(i,active_set[i]);
				// or Q.swap_index(i,active_set[i]);
	}*/

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n",iter);

	free(this->p);
	free(this->y);
	free(this->alpha);
	free(this->alpha_status);
	free(this->active_set);
	free(this->G);
	free(this->G_bar);
}

// return 1 if already optimal, return 0 otherwise
int Solver__select_working_set(Solver* this, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, int *out_i, int *out_j);
int Solver_NU__select_working_set(Solver* this, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, int *out_i, int *out_j);
int Solver_virtual__select_working_set(Solver *solver, Solver_NU *solver_nu, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, int *out_i, int *out_j) {
    if(solver_nu) {
        return Solver_NU__select_working_set(solver, svc_q, one_class_q, svr_q, out_i, out_j);
    }
    return Solver__select_working_set(solver, svc_q, one_class_q, svr_q, out_i, out_j);
}
int Solver__select_working_set(Solver* this, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, int *out_i, int *out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<this->active_size;t++)
		if(this->y[t]==+1)
		{
			if(!Solver__is_upper_bound(this, t))
				if(-this->G[t] >= Gmax)
				{
					Gmax = -this->G[t];
					Gmax_idx = t;
				}
		}
		else
		{
			if(!Solver__is_lower_bound(this, t))
				if(this->G[t] >= Gmax)
				{
					Gmax = this->G[t];
					Gmax_idx = t;
				}
		}

	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i = QMatrix_virtual__get_Q(svc_q, one_class_q, svr_q, i,this->active_size);

	for(int j=0;j<this->active_size;j++)
	{
		if(this->y[j]==+1)
		{
			if (!Solver__is_lower_bound(this, j))
			{
				double grad_diff=Gmax+this->G[j];
				if (this->G[j] >= Gmax2)
					Gmax2 = this->G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = this->QD[i]+this->QD[j]-2.0*this->y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!Solver__is_upper_bound(this, j))
			{
				double grad_diff= Gmax-this->G[j];
				if (-this->G[j] >= Gmax2)
					Gmax2 = -this->G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = this->QD[i]+this->QD[j]+2.0*this->y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(Gmax+Gmax2 < this->eps || Gmin_idx == -1)
		return 1;

	*out_i = Gmax_idx;
	*out_j = Gmin_idx;
	return 0;
}

bool Solver__be_shrunk(Solver* this, int i, double Gmax1, double Gmax2)
{
	if(Solver__is_upper_bound(this, i))
	{
		if(this->y[i]==+1)
			return(-this->G[i] > Gmax1);
		else
			return(-this->G[i] > Gmax2);
	}
	else if(Solver__is_lower_bound(this, i))
	{
		if(this->y[i]==+1)
			return(this->G[i] > Gmax2);
		else
			return(this->G[i] > Gmax1);
	}
	else
		return(false);
}

void Solver__do_shrinking(Solver* this, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q);
void Solver_NU__do_shrinking(Solver* this, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q);
void Solver_virtual__do_shrinking(Solver *solver, Solver_NU *solver_nu, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q) {
    if(solver_nu) {
        return Solver_NU__do_shrinking(solver, svc_q, one_class_q, svr_q);
    }
    return Solver__do_shrinking(solver, svc_q, one_class_q, svr_q);
}
void Solver__do_shrinking(Solver* this, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q)
{
	int i;
	double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<this->active_size;i++)
	{
		if(this->y[i]==+1)
		{
			if(!Solver__is_upper_bound(this, i))
			{
				if(-this->G[i] >= Gmax1)
					Gmax1 = -this->G[i];
			}
			if(!Solver__is_lower_bound(this, i))
			{
				if(this->G[i] >= Gmax2)
					Gmax2 = this->G[i];
			}
		}
		else
		{
			if(!Solver__is_upper_bound(this, i))
			{
				if(-this->G[i] >= Gmax2)
					Gmax2 = -this->G[i];
			}
			if(!Solver__is_lower_bound(this, i))
			{
				if(this->G[i] >= Gmax1)
					Gmax1 = this->G[i];
			}
		}
	}

	if(this->unshrink == false && Gmax1 + Gmax2 <= this->eps*10)
	{
        this->unshrink = true;
        Solver__reconstruct_gradient(this, svc_q, one_class_q, svr_q);
        this->active_size = this->l;
		info("*");
	}

	for(i=0;i<this->active_size;i++)
		if (Solver__be_shrunk(this, i, Gmax1, Gmax2))
		{
            this->active_size--;
			while (this->active_size > i)
			{
				if (!Solver__be_shrunk(this, this->active_size, Gmax1, Gmax2))
				{
					Solver__swap_index(this, svc_q, one_class_q, svr_q, i,this->active_size);
					break;
				}
                this->active_size--;
			}
		}
}

double Solver__calculate_rho(Solver* this);
double Solver_NU__calculate_rho(Solver_NU* this);
double Solver_virtual__calculate_rho(Solver *solver, Solver_NU *solver_nu) {
    if(solver_nu) {
        return Solver_NU__calculate_rho(solver_nu);
    }
    return Solver__calculate_rho(solver);
}
double Solver__calculate_rho(Solver* this)
{
	double r;
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for(int i=0;i<this->active_size;i++)
	{
		double yG = this->y[i]*this->G[i];

		if(Solver__is_upper_bound(this, i))
		{
			if(this->y[i]==-1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else if(Solver__is_lower_bound(this, i))
		{
			if(this->y[i]==+1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else
		{
			++nr_free;
			sum_free += yG;
		}
	}

	if(nr_free>0)
		r = sum_free/nr_free;
	else
		r = (ub+lb)/2;

	return r;
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
struct Solver_NU_extends_Solver_Struct
{
    Solver *super;
//public:
//	Solver_NU() {}
//	void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
//		   double *alpha, double Cp, double Cn, double eps,
//		   SolutionInfo* si, int shrinking)
//	{
//		this->si = si;
//		Solver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
//	}
//private:
	SolutionInfo *si;
//	int select_working_set(int &i, int &j);
//	double calculate_rho();
//	bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
//	void do_shrinking();
};

void Solver_NU__Solve(Solver_NU* this, int l, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, const double *p, const schar *y,
		   double *alpha, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking) {
    Solver* thisSolver = this->super;

    this->si = si;
    Solver__Solve(thisSolver, this, l,svc_q, one_class_q, svr_q,p,y,alpha,Cp,Cn,eps,si,shrinking);
}

// return 1 if already optimal, return 0 otherwise
int Solver_NU__select_working_set(Solver* this, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q, int *out_i, int *out_j)
{
	// return i,j such that y_i = y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmaxp = -INF;
	double Gmaxp2 = -INF;
	int Gmaxp_idx = -1;

	double Gmaxn = -INF;
	double Gmaxn2 = -INF;
	int Gmaxn_idx = -1;

	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<this->active_size;t++)
		if(this->y[t]==+1)
		{
			if(!Solver__is_upper_bound(this, t))
				if(-this->G[t] >= Gmaxp)
				{
					Gmaxp = -this->G[t];
					Gmaxp_idx = t;
				}
		}
		else
		{
			if(!Solver__is_lower_bound(this, t))
				if(this->G[t] >= Gmaxn)
				{
					Gmaxn = this->G[t];
					Gmaxn_idx = t;
				}
		}

	int ip = Gmaxp_idx;
	int in = Gmaxn_idx;
	const Qfloat *Q_ip = NULL;
	const Qfloat *Q_in = NULL;
	if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
		Q_ip = QMatrix_virtual__get_Q(svc_q, one_class_q, svr_q, ip,this->active_size);
	if(in != -1)
		Q_in = QMatrix_virtual__get_Q(svc_q, one_class_q, svr_q, in,this->active_size);

	for(int j=0;j<this->active_size;j++)
	{
		if(this->y[j]==+1)
		{
			if (!Solver__is_lower_bound(this, j))
			{
				double grad_diff=Gmaxp+this->G[j];
				if (this->G[j] >= Gmaxp2)
					Gmaxp2 = this->G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = this->QD[ip]+this->QD[j]-2*Q_ip[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!Solver__is_upper_bound(this, j))
			{
				double grad_diff=Gmaxn-this->G[j];
				if (-this->G[j] >= Gmaxn2)
					Gmaxn2 = -this->G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = this->QD[in]+this->QD[j]-2*Q_in[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < this->eps || Gmin_idx == -1)
		return 1;

	if (this->y[Gmin_idx] == +1)
		*out_i = Gmaxp_idx;
	else
		*out_i = Gmaxn_idx;
	*out_j = Gmin_idx;

	return 0;
}

bool Solver_NU__be_shrunk(Solver* this, int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
	if(Solver__is_upper_bound(this, i))
	{
		if(this->y[i]==+1)
			return(-this->G[i] > Gmax1);
		else
			return(-this->G[i] > Gmax4);
	}
	else if(Solver__is_lower_bound(this, i))
	{
		if(this->y[i]==+1)
			return(this->G[i] > Gmax2);
		else
			return(this->G[i] > Gmax3);
	}
	else
		return(false);
}

void Solver_NU__do_shrinking(Solver* this, SVC_Q *svc_q, ONE_CLASS_Q *one_class_q, SVR_Q *svr_q)
{
	double Gmax1 = -INF;	// max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
	double Gmax2 = -INF;	// max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
	double Gmax3 = -INF;	// max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
	double Gmax4 = -INF;	// max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

	// find maximal violating pair first
	int i;
	for(i=0;i<this->active_size;i++)
	{
		if(!Solver__is_upper_bound(this, i))
		{
			if(this->y[i]==+1)
			{
				if(-this->G[i] > Gmax1) Gmax1 = -this->G[i];
			}
			else	if(-this->G[i] > Gmax4) Gmax4 = -this->G[i];
		}
		if(!Solver__is_lower_bound(this, i))
		{
			if(this->y[i]==+1)
			{
				if(this->G[i] > Gmax2) Gmax2 = this->G[i];
			}
			else	if(this->G[i] > Gmax3) Gmax3 = this->G[i];
		}
	}

	if(this->unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= this->eps*10)
	{
        this->unshrink = true;
        Solver__reconstruct_gradient(this, svc_q, one_class_q, svr_q);
        this->active_size = this->l;
	}

	for(i=0;i<this->active_size;i++)
		if (Solver_NU__be_shrunk(this, i, Gmax1, Gmax2, Gmax3, Gmax4))
		{
            this->active_size--;
			while (this->active_size > i)
			{
				if (!Solver_NU__be_shrunk(this, this->active_size, Gmax1, Gmax2, Gmax3, Gmax4))
				{
                    Solver__swap_index(this, svc_q, one_class_q, svr_q, i,this->active_size);
					break;
				}
                this->active_size--;
			}
		}
}

double Solver_NU__calculate_rho(Solver_NU* this)
{
	int nr_free1 = 0,nr_free2 = 0;
	double ub1 = INF, ub2 = INF;
	double lb1 = -INF, lb2 = -INF;
	double sum_free1 = 0, sum_free2 = 0;

	for(int i=0;i<this->super->active_size;i++)
	{
		if(this->super->y[i]==+1)
		{
			if(Solver__is_upper_bound(this->super, i))
				lb1 = max(lb1,this->super->G[i]);
			else if(Solver__is_lower_bound(this->super, i))
				ub1 = min(ub1,this->super->G[i]);
			else
			{
				++nr_free1;
				sum_free1 += this->super->G[i];
			}
		}
		else
		{
			if(Solver__is_upper_bound(this->super, i))
				lb2 = max(lb2,this->super->G[i]);
			else if(Solver__is_lower_bound(this->super, i))
				ub2 = min(ub2,this->super->G[i]);
			else
			{
				++nr_free2;
				sum_free2 += this->super->G[i];
			}
		}
	}

	double r1,r2;
	if(nr_free1 > 0)
		r1 = sum_free1/nr_free1;
	else
		r1 = (ub1+lb1)/2;

	if(nr_free2 > 0)
		r2 = sum_free2/nr_free2;
	else
		r2 = (ub2+lb2)/2;

	this->si->r = (r1+r2)/2;
	return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
struct SVC_Q_extends_Kernel_Struct
{
    Kernel *superKernel;
//public:
//	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
//	:Kernel(prob.l, prob.x, param)
//	{
//		clone(y,y_,prob.l);
//		cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
//		QD = new double[prob.l];
//		for(int i=0;i<prob.l;i++)
//			QD[i] = (this->*kernel_function)(i,i);
//	}
//
//	Qfloat *get_Q(int i, int len) const
//	{
//		Qfloat *data;
//		int start, j;
//		if((start = cache->get_data(i,&data,len)) < len)
//		{
//#ifdef _OPENMP
//#pragma omp parallel for private(j) schedule(guided)
//#endif
//			for(j=start;j<len;j++)
//				data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
//		}
//		return data;
//	}
//
//	double *get_QD() const
//	{
//		return QD;
//	}
//
//	void swap_index(int i, int j) const
//	{
//		cache->swap_index(i,j);
//		Kernel::swap_index(i,j);
//		swap(y[i],y[j]);
//		swap(QD[i],QD[j]);
//	}

//	~SVC_Q()
//	{
//		delete[] y;
//		delete cache;
//		delete[] QD;
//	}
//private:
	schar *y;
	Cache *cache;
	double *QD;
};

SVC_Q* SVC_Q__new(const svm_problem* prob, const svm_parameter* param, const schar *y_) {
    Kernel* thisKernel = Kernel__new(prob->l, prob->x, *param);

    SVC_Q* this = malloc(sizeof(SVC_Q));this->superKernel=thisKernel;
    clone(this->y,y_,prob->l);
    this->cache = Cache__new(prob->l,(long int)(param->cache_size*(1<<20)));
    this->QD = malloc(sizeof(double) * prob->l);
    for(int i=0;i<prob->l;i++)
        this->QD[i] = (*thisKernel->kernel_function)(thisKernel, i,i);

    return this;
}

Qfloat *SVC_Q__get_Q(SVC_Q* this, int i, int len) {
    Kernel* thisKernel = this->superKernel;

    Qfloat *data;
    int start, j;
    if((start = Cache__get_data(this->cache, i,&data,len)) < len)
    {
#ifdef _OPENMP
#pragma omp parallel for private(j) schedule(guided)
#endif
        for(j=start;j<len;j++)
            data[j] = (Qfloat)(this->y[i]*this->y[j]*(*thisKernel->kernel_function)(thisKernel, i,j));
    }
    return data;
}

double *SVC_Q__get_QD(SVC_Q* this) {
    return this->QD;
}

void SVC_Q__swap_index(SVC_Q* this, int i, int j)
{
    Kernel* thisKernel = this->superKernel;

    Cache__swap_index(this->cache, i,j);
    Kernel__swap_index(thisKernel, i,j);
    swap(this->y[i],this->y[j]);
    swap(this->QD[i],this->QD[j]);
}

void SVC_Q__destructor(SVC_Q* this) {
    free(this->y);
    free(this->cache);
    free(this->QD);
}


struct ONE_CLASS_Q_extends_Kernel_Struct
{
    Kernel *superKernel;
//public:
//	ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
//	:Kernel(prob.l, prob.x, param)
//	{
//		cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
//		QD = new double[prob.l];
//		for(int i=0;i<prob.l;i++)
//			QD[i] = (this->*kernel_function)(i,i);
//	}
//
//	Qfloat *get_Q(int i, int len) const
//	{
//		Qfloat *data;
//		int start, j;
//		if((start = cache->get_data(i,&data,len)) < len)
//		{
//			for(j=start;j<len;j++)
//				data[j] = (Qfloat)(this->*kernel_function)(i,j);
//		}
//		return data;
//	}
//
//	double *get_QD() const
//	{
//		return QD;
//	}

//	void swap_index(int i, int j) const
//	{
//		cache->swap_index(i,j);
//		Kernel::swap_index(i,j);
//		swap(QD[i],QD[j]);
//	}

//	~ONE_CLASS_Q()
//	{
//		delete cache;
//		delete[] QD;
//	}
//private:
	Cache *cache;
	double *QD;
};

ONE_CLASS_Q* ONE_CLASS_Q__new(const svm_problem* prob, const svm_parameter* param) {
    Kernel* thisKernel = Kernel__new(prob->l, prob->x, *param);

    ONE_CLASS_Q* this = malloc(sizeof(ONE_CLASS_Q));this->superKernel=thisKernel;
    this->cache = Cache__new(prob->l,(long int)(param->cache_size*(1<<20)));
    this->QD = Malloc(double, prob->l);
    for(int i=0;i<prob->l;i++)
        this->QD[i] = (*thisKernel->kernel_function)(thisKernel, i,i);

    return this;
}

Qfloat *ONE_CLASS_Q__get_Q(ONE_CLASS_Q* this, int i, int len)
{
    Kernel* thisKernel = this->superKernel;

    Qfloat *data;
    int start, j;
    if((start = Cache__get_data(this->cache, i,&data,len)) < len)
    {
        for(j=start;j<len;j++)
            data[j] = (Qfloat)(*thisKernel->kernel_function)(thisKernel, i,j);
    }
    return data;
}

double *ONE_CLASS_Q__get_QD(ONE_CLASS_Q* this)
{
    return this->QD;
}

void ONE_CLASS_Q__swap_index(ONE_CLASS_Q* this, int i, int j) {
    Kernel* thisKernel = this->superKernel;

    Cache__swap_index(this->cache, i,j);
    Kernel__swap_index(thisKernel, i,j);
    swap(this->QD[i],this->QD[j]);
}

void ONE_CLASS_Q__destructor(ONE_CLASS_Q* this) {
    free(this->cache);
    free(this->QD);
}


struct SVR_Q_extends_Kernel_Struct
{
    Kernel *superKernel;
//public:
//	SVR_Q(const svm_problem& prob, const svm_parameter& param)
//	:Kernel(prob.l, prob.x, param)
//	{
//		l = prob.l;
//		cache = new Cache(l,(long int)(param.cache_size*(1<<20)));
//		QD = new double[2*l];
//		sign = new schar[2*l];
//		index = new int[2*l];
//		for(int k=0;k<l;k++)
//		{
//			sign[k] = 1;
//			sign[k+l] = -1;
//			index[k] = k;
//			index[k+l] = k;
//			QD[k] = (this->*kernel_function)(k,k);
//			QD[k+l] = QD[k];
//		}
//		buffer[0] = new Qfloat[2*l];
//		buffer[1] = new Qfloat[2*l];
//		next_buffer = 0;
//	}

//	void swap_index(int i, int j) const
//	{
//		swap(sign[i],sign[j]);
//		swap(index[i],index[j]);
//		swap(QD[i],QD[j]);
//	}

//	Qfloat *get_Q(int i, int len) const
//	{
//		Qfloat *data;
//		int j, real_i = index[i];
//		if(cache->get_data(real_i,&data,l) < l)
//		{
//#ifdef _OPENMP
//#pragma omp parallel for private(j) schedule(guided)
//#endif
//			for(j=0;j<l;j++)
//				data[j] = (Qfloat)(this->*kernel_function)(real_i,j);
//		}
//
//		// reorder and copy
//		Qfloat *buf = buffer[next_buffer];
//		next_buffer = 1 - next_buffer;
//		schar si = sign[i];
//		for(j=0;j<len;j++)
//			buf[j] = (Qfloat) si * (Qfloat) sign[j] * data[index[j]];
//		return buf;
//	}

//	double *get_QD() const
//	{
//		return QD;
//	}

//	~SVR_Q()
//	{
//		delete cache;
//		delete[] sign;
//		delete[] index;
//		delete[] buffer[0];
//		delete[] buffer[1];
//		delete[] QD;
//	}
//private:
	int l;
	Cache *cache;
	schar *sign;
	int *index;
    int next_buffer;
	Qfloat *buffer[2];
	double *QD;
};

SVR_Q* SVR_Q__new(const svm_problem* prob, const svm_parameter* param)
{
    Kernel* thisKernel = Kernel__new(prob->l, prob->x, *param);

    SVR_Q* this = malloc(sizeof(SVR_Q));this->superKernel=thisKernel;
    this->l = prob->l;
    this->cache = Cache__new(this->l,(long int)(param->cache_size*(1<<20)));
    this->QD = Malloc(double, 2*this->l);
    this->sign = Malloc(schar, 2*this->l);
    this->index = Malloc(int, 2*this->l);
    for(int k=0;k<this->l;k++)
    {
        this->sign[k] = 1;
        this->sign[k+this->l] = -1;
        this->index[k] = k;
        this->index[k+this->l] = k;
        this->QD[k] = (*thisKernel->kernel_function)(thisKernel, k,k);
        this->QD[k+this->l] = this->QD[k];
    }
    this->buffer[0] = Malloc(Qfloat, 2*this->l);
    this->buffer[1] = Malloc(Qfloat, 2*this->l);
    this->next_buffer = 0;

    return this;
}

void SVR_Q__swap_index(SVR_Q* this, int i, int j)
{
    swap(this->sign[i],this->sign[j]);
    swap(this->index[i],this->index[j]);
    swap(this->QD[i],this->QD[j]);
}

Qfloat *SVR_Q__get_Q(SVR_Q* this, int i, int len)
{
    Kernel* thisKernel = this->superKernel;

    Qfloat *data;
    int j, real_i = this->index[i];
    if(Cache__get_data(this->cache, real_i,&data,this->l) < this->l)
{
#ifdef _OPENMP
#pragma omp parallel for private(j) schedule(guided)
#endif
    for(j=0;j<this->l;j++)
    data[j] = (Qfloat)(*thisKernel->kernel_function)(thisKernel, real_i,j);
    }

    // reorder and copy
    Qfloat *buf = this->buffer[this->next_buffer];
    this->next_buffer = 1 - this->next_buffer;
    schar si = this->sign[i];
    for(j=0;j<len;j++)
        buf[j] = (Qfloat) si * (Qfloat) this->sign[j] * data[this->index[j]];
    return buf;
}

double *SVR_Q__get_QD(SVR_Q* this)
{
    return this->QD;
}

void SVR_Q__destructor(SVR_Q* this)
{
    free(this->cache);
    free(this->sign);
    free(this->index);
    free(this->buffer[0]);
    free(this->buffer[1]);
    free(this->QD);
}

//
// construct and solve various formulations
//
typedef SolutionInfo Solver__SolutionInfo;
static void solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver__SolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;
	double *minus_ones = Malloc(double, l);
	schar *y = Malloc(schar, l);

	int i;

	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		minus_ones[i] = -1;
		if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
	}

    SVC_Q* svc_q = SVC_Q__new(prob,param,y);
	Solver s;
	Solver__Solve(&s, NULL, l, svc_q, NULL, NULL, minus_ones, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking);

	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		info("nu = %f\n", sum_alpha/(Cp*prob->l));

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	free(minus_ones);
    free(y);
}

static void solve_nu_svc(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver__SolutionInfo* si)
{
	int i;
	int l = prob->l;
	double nu = param->nu;

	schar *y = Malloc(schar, l);

	for(i=0;i<l;i++)
		if(prob->y[i]>0)
			y[i] = +1;
		else
			y[i] = -1;

	double sum_pos = nu*l/2;
	double sum_neg = nu*l/2;

	for(i=0;i<l;i++)
		if(y[i] == +1)
		{
			alpha[i] = min(1.0,sum_pos);
			sum_pos -= alpha[i];
		}
		else
		{
			alpha[i] = min(1.0,sum_neg);
			sum_neg -= alpha[i];
		}

	double *zeros = Malloc(double, l);

	for(i=0;i<l;i++)
		zeros[i] = 0;

    SVC_Q* svc_q = SVC_Q__new(prob,param,y);
	Solver_NU s;s.super = Malloc(Solver, 1);
    Solver_NU__Solve(&s, l, svc_q, NULL, NULL, zeros, y,
		alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
	double r = si->r;

	info("C = %f\n",1/r);

	for(i=0;i<l;i++)
		alpha[i] *= y[i]/r;

	si->rho /= r;
	si->obj /= (r*r);
	si->upper_bound_p = 1/r;
	si->upper_bound_n = 1/r;

	free(y);
    free(zeros);
}

static void solve_one_class(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver__SolutionInfo* si)
{
	int l = prob->l;
	double *zeros = Malloc(double, l);
	schar *ones = Malloc(schar, l);
	int i;

	int n = (int)(param->nu*prob->l);	// # of alpha's at upper bound

	for(i=0;i<n;i++)
		alpha[i] = 1;
	if(n<prob->l)
		alpha[n] = param->nu * prob->l - n;
	for(i=n+1;i<l;i++)
		alpha[i] = 0;

	for(i=0;i<l;i++)
	{
		zeros[i] = 0;
		ones[i] = 1;
	}

    ONE_CLASS_Q* one_class_q = ONE_CLASS_Q__new(prob,param);
	Solver s;
    Solver__Solve(&s, NULL, l, NULL, one_class_q, NULL, zeros, ones,
		alpha, 1.0, 1.0, param->eps, si, param->shrinking);

	free(zeros);
	free(ones);
}

static void solve_epsilon_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver__SolutionInfo* si)
{
	int l = prob->l;
	double *alpha2 = Malloc(double, 2*l);
	double *linear_term = Malloc(double, 2*l);
	schar *y = Malloc(schar, 2*l);
	int i;

	for(i=0;i<l;i++)
	{
		alpha2[i] = 0;
		linear_term[i] = param->p - prob->y[i];
		y[i] = 1;

		alpha2[i+l] = 0;
		linear_term[i+l] = param->p + prob->y[i];
		y[i+l] = -1;
	}

    SVR_Q* svr_q = SVR_Q__new(prob,param);
	Solver s;
    Solver__Solve(&s, NULL, 2*l, NULL, NULL, svr_q, linear_term, y,
		alpha2, param->C, param->C, param->eps, si, param->shrinking);

	double sum_alpha = 0;
	for(i=0;i<l;i++)
	{
		alpha[i] = alpha2[i] - alpha2[i+l];
		sum_alpha += fabs(alpha[i]);
	}
	info("nu = %f\n",sum_alpha/(param->C*l));

	free(alpha2);
    free(linear_term);
    free(y);
}

static void solve_nu_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver__SolutionInfo* si)
{
	int l = prob->l;
	double C = param->C;
	double *alpha2 = Malloc(double, 2*l);
	double *linear_term = Malloc(double, 2*l);
	schar *y = Malloc(schar, 2*l);
	int i;

	double sum = C * param->nu * l / 2;
	for(i=0;i<l;i++)
	{
		alpha2[i] = alpha2[i+l] = min(sum,C);
		sum -= alpha2[i];

		linear_term[i] = - prob->y[i];
		y[i] = 1;

		linear_term[i+l] = prob->y[i];
		y[i+l] = -1;
	}

    SVR_Q* svr_q = SVR_Q__new(prob,param);
	Solver_NU s;s.super = Malloc(Solver, 1);
    Solver_NU__Solve(&s, 2*l, NULL, NULL, svr_q, linear_term, y,
		alpha2, C, C, param->eps, si, param->shrinking);

	info("epsilon = %f\n",-si->r);

	for(i=0;i<l;i++)
		alpha[i] = alpha2[i] - alpha2[i+l];

    free(alpha2);
    free(linear_term);
    free(y);
}

//
// decision_function
//
typedef struct decision_function_Struct decision_function;
struct decision_function_Struct
{
	double *alpha;
	double rho;
};

static decision_function svm_train_one(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn)
{
	double *alpha = Malloc(double,prob->l);
	Solver__SolutionInfo si;
	switch(param->svm_type)
	{
		case C_SVC:
			solve_c_svc(prob,param,alpha,&si,Cp,Cn);
			break;
		case NU_SVC:
			solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
			solve_one_class(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			solve_nu_svr(prob,param,alpha,&si);
			break;
	}

	info("obj = %f, rho = %f\n",si.obj,si.rho);

	// output SVs

	int nSV = 0;
	int nBSV = 0;
	for(int i=0;i<prob->l;i++)
	{
		if(fabs(alpha[i]) > 0)
		{
			++nSV;
			if(prob->y[i] > 0)
			{
				if(fabs(alpha[i]) >= si.upper_bound_p)
					++nBSV;
			}
			else
			{
				if(fabs(alpha[i]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}

	info("nSV = %d, nBSV = %d\n",nSV,nBSV);

	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	return f;
}

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
static void sigmoid_train(
	int l, const double *dec_values, const double *labels,
	double* A, double* B)
{
	double prior1=0, prior0 = 0;
	int i;

	for (i=0;i<l;i++)
		if (labels[i] > 0) prior1+=1;
		else prior0+=1;

	int max_iter=100;	// Maximal number of iterations
	double min_step=1e-10;	// Minimal step taken in line search
	double sigma=1e-12;	// For numerically strict PD of Hessian
	double eps=1e-5;
	double hiTarget=(prior1+1.0)/(prior1+2.0);
	double loTarget=1/(prior0+2.0);
	double *t=Malloc(double,l);
	double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
	double newA,newB,newf,d1,d2;
	int iter;

	// Initial Point and Initial Fun Value
	*A=0.0; *B=log((prior0+1.0)/(prior1+1.0));
	double fval = 0.0;

	for (i=0;i<l;i++)
	{
		if (labels[i]>0) t[i]=hiTarget;
		else t[i]=loTarget;
		fApB = dec_values[i]*(*A)+(*B);
		if (fApB>=0)
			fval += t[i]*fApB + log(1+exp(-fApB));
		else
			fval += (t[i] - 1)*fApB +log(1+exp(fApB));
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// Update Gradient and Hessian (use H' = H + sigma I)
		h11=sigma; // numerically ensures strict PD
		h22=sigma;
		h21=0.0;g1=0.0;g2=0.0;
		for (i=0;i<l;i++)
		{
			fApB = dec_values[i]*(*A)+(*B);
			if (fApB >= 0)
			{
				p=exp(-fApB)/(1.0+exp(-fApB));
				q=1.0/(1.0+exp(-fApB));
			}
			else
			{
				p=1.0/(1.0+exp(fApB));
				q=exp(fApB)/(1.0+exp(fApB));
			}
			d2=p*q;
			h11+=dec_values[i]*dec_values[i]*d2;
			h22+=d2;
			h21+=dec_values[i]*d2;
			d1=t[i]-p;
			g1+=dec_values[i]*d1;
			g2+=d1;
		}

		// Stopping Criteria
		if (fabs(g1)<eps && fabs(g2)<eps)
			break;

		// Finding Newton direction: -inv(H') * g
		det=h11*h22-h21*h21;
		dA=-(h22*g1 - h21 * g2) / det;
		dB=-(-h21*g1+ h11 * g2) / det;
		gd=g1*dA+g2*dB;


		stepsize = 1;		// Line Search
		while (stepsize >= min_step)
		{
			newA = *A + stepsize * dA;
			newB = *B + stepsize * dB;

			// New function value
			newf = 0.0;
			for (i=0;i<l;i++)
			{
				fApB = dec_values[i]*newA+newB;
				if (fApB >= 0)
					newf += t[i]*fApB + log(1+exp(-fApB));
				else
					newf += (t[i] - 1)*fApB +log(1+exp(fApB));
			}
			// Check sufficient decrease
			if (newf<fval+0.0001*stepsize*gd)
			{
                *A=newA;*B=newB;fval=newf;
				break;
			}
			else
				stepsize = stepsize / 2.0;
		}

		if (stepsize < min_step)
		{
			info("Line search fails in two-class probability estimates\n");
			break;
		}
	}

	if (iter>=max_iter)
		info("Reaching maximal iterations in two-class probability estimates\n");
	free(t);
}

static double sigmoid_predict(double decision_value, double A, double B)
{
	double fApB = decision_value*A+B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
		return exp(-fApB)/(1.0+exp(-fApB));
	else
		return 1.0/(1+exp(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng to predict probabilities
static void multiclass_probability(int k, double **r, double *p)
{
	int t,j;
	int iter = 0, max_iter=max(100,k);
	double **Q=Malloc(double *,k);
	double *Qp=Malloc(double,k);
	double pQp, eps=0.005/k;

	for (t=0;t<k;t++)
	{
		p[t]=1.0/k;  // Valid if k = 1
		Q[t]=Malloc(double,k);
		Q[t][t]=0;
		for (j=0;j<t;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=Q[j][t];
		}
		for (j=t+1;j<k;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=-r[j][t]*r[t][j];
		}
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp=0;
		for (t=0;t<k;t++)
		{
			Qp[t]=0;
			for (j=0;j<k;j++)
				Qp[t]+=Q[t][j]*p[j];
			pQp+=p[t]*Qp[t];
		}
		double max_error=0;
		for (t=0;t<k;t++)
		{
			double error=fabs(Qp[t]-pQp);
			if (error>max_error)
				max_error=error;
		}
		if (max_error<eps) break;

		for (t=0;t<k;t++)
		{
			double diff=(-Qp[t]+pQp)/Q[t][t];
			p[t]+=diff;
			pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
			for (j=0;j<k;j++)
			{
				Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
				p[j]/=(1+diff);
			}
		}
	}
	if (iter>=max_iter)
		info("Exceeds max_iter in multiclass_prob\n");
	for(t=0;t<k;t++) free(Q[t]);
	free(Q);
	free(Qp);
}

// Using cross-validation decision values to get parameters for SVC probability estimates
static void svm_binary_svc_probability(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn, double* probA, double* probB)
{
	int i;
	int nr_fold = 5;
	int *perm = Malloc(int,prob->l);
	double *dec_values = Malloc(double,prob->l);

	// random shuffle
	for(i=0;i<prob->l;i++) perm[i]=i;
	for(i=0;i<prob->l;i++)
	{
		int j = i+rand()%(prob->l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<nr_fold;i++)
	{
		int begin = i*prob->l/nr_fold;
		int end = (i+1)*prob->l/nr_fold;
		int j,k;
		struct svm_problem subprob;

		subprob.l = prob->l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<prob->l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		int p_count=0,n_count=0;
		for(j=0;j<k;j++)
			if(subprob.y[j]>0)
				p_count++;
			else
				n_count++;

		if(p_count==0 && n_count==0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 0;
		else if(p_count > 0 && n_count == 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 1;
		else if(p_count == 0 && n_count > 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = -1;
		else
		{
			svm_parameter subparam = *param;
			subparam.probability=0;
			subparam.C=1.0;
			subparam.nr_weight=2;
			subparam.weight_label = Malloc(int,2);
			subparam.weight = Malloc(double,2);
			subparam.weight_label[0]=+1;
			subparam.weight_label[1]=-1;
			subparam.weight[0]=Cp;
			subparam.weight[1]=Cn;
			struct svm_model *submodel = svm_train(&subprob,&subparam);
			for(j=begin;j<end;j++)
			{
				svm_predict_values(submodel,prob->x[perm[j]],&(dec_values[perm[j]]));
				// ensure +1 -1 order; reason not using CV subroutine
				dec_values[perm[j]] *= submodel->label[0];
			}
			svm_free_and_destroy_model(&submodel);
			svm_destroy_param(&subparam);
		}
		free(subprob.x);
		free(subprob.y);
	}
	sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
	free(dec_values);
	free(perm);
}

// Binning method from the oneclass_prob paper by Que and Lin to predict the probability as a normal instance (i.e., not an outlier)
static double predict_one_class_probability(const svm_model *model, double dec_value)
{
	double prob_estimate = 0.0;
	int nr_marks = 10;

	if(dec_value < model->prob_density_marks[0])
		prob_estimate = 0.001;
	else if(dec_value > model->prob_density_marks[nr_marks-1])
		prob_estimate = 0.999;
	else
	{
		for(int i=1;i<nr_marks;i++)
			if(dec_value < model->prob_density_marks[i])
			{
				prob_estimate = (double)i/nr_marks;
				break;
			}
	}
	return prob_estimate;
}

static int compare_double(const void *a, const void *b)
{
	if(*(double *)a > *(double *)b)
		return 1;
	else if(*(double *)a < *(double *)b)
		return -1;
	return 0;
}

// Get parameters for one-class SVM probability estimates
static int svm_one_class_probability(const svm_problem *prob, const svm_model *model, double *prob_density_marks)
{
	double *dec_values = Malloc(double,prob->l);
	double *pred_results = Malloc(double,prob->l);
	int ret = 0;
	int nr_marks = 10;

	for(int i=0;i<prob->l;i++)
		pred_results[i] = svm_predict_values(model,prob->x[i],&dec_values[i]);
	qsort(dec_values,prob->l,sizeof(double),compare_double);

	int neg_counter=0;
	for(int i=0;i<prob->l;i++)
		if(dec_values[i]>=0)
		{
			neg_counter = i;
			break;
		}

	int pos_counter = prob->l-neg_counter;
	if(neg_counter<nr_marks/2 || pos_counter<nr_marks/2)
	{
		fprintf(stderr,"WARNING: number of positive or negative decision values <%d; too few to do a probability estimation.\n",nr_marks/2);
		ret = -1;
	}
	else
	{
		// Binning by density
		double *tmp_marks = Malloc(double,nr_marks+1);
		int mid = nr_marks/2;
		for(int i=0;i<mid;i++)
			tmp_marks[i] = dec_values[i*neg_counter/mid];
		tmp_marks[mid] = 0;
		for(int i=mid+1;i<nr_marks+1;i++)
			tmp_marks[i] = dec_values[neg_counter-1+(i-mid)*pos_counter/mid];

		for(int i=0;i<nr_marks;i++)
			prob_density_marks[i] = (tmp_marks[i]+tmp_marks[i+1])/2;
		free(tmp_marks);
	}
	free(dec_values);
	free(pred_results);
	return ret;
}

// Return parameter of a Laplace distribution
static double svm_svr_probability(
	const svm_problem *prob, const svm_parameter *param)
{
	int i;
	int nr_fold = 5;
	double *ymv = Malloc(double,prob->l);
	double mae = 0;

	svm_parameter newparam = *param;
	newparam.probability = 0;
	svm_cross_validation(prob,&newparam,nr_fold,ymv);
	for(i=0;i<prob->l;i++)
	{
		ymv[i]=prob->y[i]-ymv[i];
		mae += fabs(ymv[i]);
	}
	mae /= prob->l;
	double std=sqrt(2*mae*mae);
	int count=0;
	mae=0;
	for(i=0;i<prob->l;i++)
		if (fabs(ymv[i]) > 5*std)
			count=count+1;
		else
			mae+=fabs(ymv[i]);
	mae /= (prob->l-count);
	info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
	free(ymv);
	return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set.
	// However, for two-class sets with -1/+1 labels and -1 appears first,
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
	svm_model *model = Malloc(svm_model,1);
	model->param = *param;
	model->free_sv = 0;	// XXX

	if(param->svm_type == ONE_CLASS ||
	   param->svm_type == EPSILON_SVR ||
	   param->svm_type == NU_SVR)
	{
		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->probA = NULL; model->probB = NULL;
		model->prob_density_marks = NULL;
		model->sv_coef = Malloc(double *,1);

		decision_function f = svm_train_one(prob,param,0,0);
		model->rho = Malloc(double,1);
		model->rho[0] = f.rho;

		int nSV = 0;
		int i;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0) ++nSV;
		model->l = nSV;
		model->SV = Malloc(svm_node *,nSV);
		model->sv_coef[0] = Malloc(double,nSV);
		model->sv_indices = Malloc(int,nSV);
		int j = 0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0)
			{
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i];
				model->sv_indices[j] = i+1;
				++j;
			}

		if(param->probability &&
		   (param->svm_type == EPSILON_SVR ||
		    param->svm_type == NU_SVR))
		{
			model->probA = Malloc(double,1);
			model->probA[0] = svm_svr_probability(prob,param);
		}
		else if(param->probability && param->svm_type == ONE_CLASS)
		{
			int nr_marks = 10;
			double *prob_density_marks = Malloc(double,nr_marks);

			if(svm_one_class_probability(prob,model,prob_density_marks) == 0)
				model->prob_density_marks = prob_density_marks;
			else
				free(prob_density_marks);
		}

		free(f.alpha);
	}
	else
	{
		// classification
		int l = prob->l;
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
		if(nr_class == 1)
			info("WARNING: training data in only one class. See README for details.\n");

		svm_node **x = Malloc(svm_node *,l);
		int i;
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		// calculate weighted C

		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{
			int j;
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models

		bool *nonzero = Malloc(bool,l);
		for(i=0;i<l;i++)
			nonzero[i] = false;
		decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

		double *probA=NULL,*probB=NULL;
		if (param->probability)
		{
			probA=Malloc(double,nr_class*(nr_class-1)/2);
			probB=Malloc(double,nr_class*(nr_class-1)/2);
		}

		int p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				svm_problem sub_prob;
				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];
				sub_prob.l = ci+cj;
				sub_prob.x = Malloc(svm_node *,sub_prob.l);
				sub_prob.y = Malloc(double,sub_prob.l);
				int k;
				for(k=0;k<ci;k++)
				{
					sub_prob.x[k] = x[si+k];
					sub_prob.y[k] = +1;
				}
				for(k=0;k<cj;k++)
				{
					sub_prob.x[ci+k] = x[sj+k];
					sub_prob.y[ci+k] = -1;
				}

				if(param->probability)
					svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],&probA[p],&probB[p]);

				f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
				for(k=0;k<ci;k++)
					if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si+k] = true;
				for(k=0;k<cj;k++)
					if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k] = true;
				free(sub_prob.x);
				free(sub_prob.y);
				++p;
			}

		// build output

		model->nr_class = nr_class;

		model->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model->label[i] = label[i];

		model->rho = Malloc(double,nr_class*(nr_class-1)/2);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			model->rho[i] = f[i].rho;

		if(param->probability)
		{
			model->probA = Malloc(double,nr_class*(nr_class-1)/2);
			model->probB = Malloc(double,nr_class*(nr_class-1)/2);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
			{
				model->probA[i] = probA[i];
				model->probB[i] = probB[i];
			}
		}
		else
		{
			model->probA=NULL;
			model->probB=NULL;
		}
		model->prob_density_marks=NULL;	// for one-class SVM probabilistic outputs only

		int total_sv = 0;
		int *nz_count = Malloc(int,nr_class);
		model->nSV = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
		{
			int nSV = 0;
			for(int j=0;j<count[i];j++)
				if(nonzero[start[i]+j])
				{
					++nSV;
					++total_sv;
				}
			model->nSV[i] = nSV;
			nz_count[i] = nSV;
		}

		info("Total nSV = %d\n",total_sv);

		model->l = total_sv;
		model->SV = Malloc(svm_node *,total_sv);
		model->sv_indices = Malloc(int,total_sv);
		p = 0;
		for(i=0;i<l;i++)
			if(nonzero[i])
			{
				model->SV[p] = x[i];
				model->sv_indices[p++] = perm[i] + 1;
			}

		int *nz_start = Malloc(int,nr_class);
		nz_start[0] = 0;
		for(i=1;i<nr_class;i++)
			nz_start[i] = nz_start[i-1]+nz_count[i-1];

		model->sv_coef = Malloc(double *,nr_class-1);
		for(i=0;i<nr_class-1;i++)
			model->sv_coef[i] = Malloc(double,total_sv);

		p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];

				int q = nz_start[i];
				int k;
				for(k=0;k<ci;k++)
					if(nonzero[si+k])
						model->sv_coef[j-1][q++] = f[p].alpha[k];
				q = nz_start[j];
				for(k=0;k<cj;k++)
					if(nonzero[sj+k])
						model->sv_coef[i][q++] = f[p].alpha[ci+k];
				++p;
			}

		free(label);
		free(probA);
		free(probB);
		free(count);
		free(perm);
		free(start);
		free(x);
		free(weighted_C);
		free(nonzero);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			free(f[i].alpha);
		free(f);
		free(nz_count);
		free(nz_start);
	}
	return model;
}

// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	int nr_class;
	if (nr_fold > l)
	{
		fprintf(stderr,"WARNING: # folds (%d) > # data (%d). Will use # folds = # data instead (i.e., leave-one-out cross validation)\n", nr_fold, l);
		nr_fold = l;
	}
	fold_start = Malloc(int,nr_fold+1);
	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if((param->svm_type == C_SVC ||
	    param->svm_type == NU_SVC) && nr_fold < l)
	{
		int *start = NULL;
		int *label = NULL;
		int *count = NULL;
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

		// random shuffle and then data grouped by fold using the array perm
		int *fold_count = Malloc(int,nr_fold);
		int c;
		int *index = Malloc(int,l);
		for(i=0;i<l;i++)
			index[i]=perm[i];
		for (c=0; c<nr_class; c++)
			for(i=0;i<count[c];i++)
			{
				int j = i+rand()%(count[c]-i);
				swap(index[start[c]+j],index[start[c]+i]);
			}
		for(i=0;i<nr_fold;i++)
		{
			fold_count[i] = 0;
			for (c=0; c<nr_class;c++)
				fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
		}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		for (c=0; c<nr_class;c++)
			for(i=0;i<nr_fold;i++)
			{
				int begin = start[c]+i*count[c]/nr_fold;
				int end = start[c]+(i+1)*count[c]/nr_fold;
				for(int j=begin;j<end;j++)
				{
					perm[fold_start[i]] = index[j];
					fold_start[i]++;
				}
			}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		free(start);
		free(label);
		free(count);
		free(index);
		free(fold_count);
	}
	else
	{
		for(i=0;i<l;i++) perm[i]=i;
		for(i=0;i<l;i++)
		{
			int j = i+rand()%(l-i);
			swap(perm[i],perm[j]);
		}
		for(i=0;i<=nr_fold;i++)
			fold_start[i]=i*l/nr_fold;
	}

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct svm_problem subprob;

		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct svm_model *submodel = svm_train(&subprob,param);
		if(param->probability &&
		   (param->svm_type == C_SVC || param->svm_type == NU_SVC))
		{
			double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
			for(j=begin;j<end;j++)
				target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
			free(prob_estimates);
		}
		else
			for(j=begin;j<end;j++)
				target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
		svm_free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}


int svm_get_svm_type(const svm_model *model)
{
	return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model)
{
	return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label)
{
	if (model->label != NULL)
		for(int i=0;i<model->nr_class;i++)
			label[i] = model->label[i];
}

void svm_get_sv_indices(const svm_model *model, int* indices)
{
	if (model->sv_indices != NULL)
		for(int i=0;i<model->l;i++)
			indices[i] = model->sv_indices[i];
}

int svm_get_nr_sv(const svm_model *model)
{
	return model->l;
}

double svm_get_svr_probability(const svm_model *model)
{
	if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
	    model->probA!=NULL)
		return model->probA[0];
	else
	{
		fprintf(stderr,"Model doesn't contain information for SVR probability inference\n");
		return 0;
	}
}

double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
	int i;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
	{
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
#ifdef _OPENMP
#pragma omp parallel for private(i) reduction(+:sum) schedule(guided)
#endif
		for(i=0;i<model->l;i++)
			sum += sv_coef[i] * Kernel__k_function(x,model->SV[i],model->param);
		sum -= model->rho[0];
		*dec_values = sum;

		if(model->param.svm_type == ONE_CLASS)
			return (sum>0)?1:-1;
		else
			return sum;
	}
	else
	{
		int nr_class = model->nr_class;
		int l = model->l;

		double *kvalue = Malloc(double,l);
#ifdef _OPENMP
#pragma omp parallel for private(i) schedule(guided)
#endif
		for(i=0;i<l;i++)
			kvalue[i] = Kernel__k_function(x,model->SV[i],model->param);

		int *start = Malloc(int,nr_class);
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+model->nSV[i-1];

		int *vote = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			vote[i] = 0;

		int p=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model->nSV[i];
				int cj = model->nSV[j];

				int k;
				double *coef1 = model->sv_coef[j-1];
				double *coef2 = model->sv_coef[i];
				for(k=0;k<ci;k++)
					sum += coef1[si+k] * kvalue[si+k];
				for(k=0;k<cj;k++)
					sum += coef2[sj+k] * kvalue[sj+k];
				sum -= model->rho[p];
				dec_values[p] = sum;

				if(dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		int vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		free(kvalue);
		free(start);
		free(vote);
		return model->label[vote_max_idx];
	}
}

double svm_predict(const svm_model *model, const svm_node *x)
{
	int nr_class = model->nr_class;
	double *dec_values;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
		dec_values = Malloc(double, 1);
	else
		dec_values = Malloc(double, nr_class*(nr_class-1)/2);
	double pred_result = svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}

double svm_predict_probability(
	const svm_model *model, const svm_node *x, double *prob_estimates)
{
	if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
	    model->probA!=NULL && model->probB!=NULL)
	{
		int i;
		int nr_class = model->nr_class;
		double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
		svm_predict_values(model, x, dec_values);

		double min_prob=1e-7;
		double **pairwise_prob=Malloc(double *,nr_class);
		for(i=0;i<nr_class;i++)
			pairwise_prob[i]=Malloc(double,nr_class);
		int k=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				pairwise_prob[i][j]=min(max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
				pairwise_prob[j][i]=1-pairwise_prob[i][j];
				k++;
			}
		if (nr_class == 2)
		{
			prob_estimates[0] = pairwise_prob[0][1];
			prob_estimates[1] = pairwise_prob[1][0];
		}
		else
			multiclass_probability(nr_class,pairwise_prob,prob_estimates);

		int prob_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(prob_estimates[i] > prob_estimates[prob_max_idx])
				prob_max_idx = i;
		for(i=0;i<nr_class;i++)
			free(pairwise_prob[i]);
		free(dec_values);
		free(pairwise_prob);
		return model->label[prob_max_idx];
	}
	else if(model->param.svm_type == ONE_CLASS && model->prob_density_marks!=NULL)
	{
		double dec_value;
		double pred_result = svm_predict_values(model,x,&dec_value);
		prob_estimates[0] = predict_one_class_probability(model,dec_value);
		prob_estimates[1] = 1-prob_estimates[0];
		return pred_result;
	}
	else
		return svm_predict(model, x);
}

static const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

int svm_save_model(const char *model_file_name, const svm_model *model)
{
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale) {
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	const svm_parameter param = model->param;

	fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
	fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

	if(param.kernel_type == POLY)
		fprintf(fp,"degree %d\n", param.degree);

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		fprintf(fp,"gamma %.17g\n", param.gamma);

	if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
		fprintf(fp,"coef0 %.17g\n", param.coef0);

	int nr_class = model->nr_class;
	int l = model->l;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n",l);

	{
		fprintf(fp, "rho");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %.17g",model->rho[i]);
		fprintf(fp, "\n");
	}

	if(model->label)
	{
		fprintf(fp, "label");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->label[i]);
		fprintf(fp, "\n");
	}

	if(model->probA) // regression has probA only
	{
		fprintf(fp, "probA");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %.17g",model->probA[i]);
		fprintf(fp, "\n");
	}
	if(model->probB)
	{
		fprintf(fp, "probB");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %.17g",model->probB[i]);
		fprintf(fp, "\n");
	}
	if(model->prob_density_marks)
	{
		fprintf(fp, "prob_density_marks");
		int nr_marks=10;
		for(int i=0;i<nr_marks;i++)
			fprintf(fp," %.17g",model->prob_density_marks[i]);
		fprintf(fp, "\n");
	}

	if(model->nSV)
	{
		fprintf(fp, "nr_sv");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->nSV[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "SV\n");
	const double * const *sv_coef = model->sv_coef;
	const svm_node * const *SV = model->SV;

	for(int i=0;i<l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			fprintf(fp, "%.17g ",sv_coef[j][i]);

		const svm_node *p = SV[i];

		if(param.kernel_type == PRECOMPUTED)
			fprintf(fp,"0:%d ",(int)(p->value));
		else
			while(p->index != -1)
			{
				fprintf(fp,"%d:%.8g ",p->index,p->value);
				p++;
			}
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var) do{ if (fscanf(_stream, _format, _var) != 1) return false; }while(0)
bool read_model_header(FILE *fp, svm_model* model)
{
	svm_parameter* param = &model->param;
	// parameters for training only won't be assigned, but arrays are assigned as NULL for safety
	param->nr_weight = 0;
	param->weight_label = NULL;
	param->weight = NULL;

	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;svm_type_table[i];i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					param->svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				return false;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					param->kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");
				return false;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			FSCANF(fp,"%d",&param->degree);
		else if(strcmp(cmd,"gamma")==0)
			FSCANF(fp,"%lf",&param->gamma);
		else if(strcmp(cmd,"coef0")==0)
			FSCANF(fp,"%lf",&param->coef0);
		else if(strcmp(cmd,"nr_class")==0)
			FSCANF(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			FSCANF(fp,"%d",&model->l);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->rho = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->rho[i]);
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label = Malloc(int,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%d",&model->label[i]);
		}
		else if(strcmp(cmd,"probA")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probA = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->probA[i]);
		}
		else if(strcmp(cmd,"probB")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probB = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->probB[i]);
		}
		else if(strcmp(cmd,"prob_density_marks")==0)
		{
			int n = 10;	// nr_marks
			model->prob_density_marks = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->prob_density_marks[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			model->nSV = Malloc(int,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%d",&model->nSV[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			return false;
		}
	}

	return true;

}

svm_model *svm_load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale) {
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	// read parameters

	svm_model *model = Malloc(svm_model,1);
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->prob_density_marks = NULL;
	model->sv_indices = NULL;
	model->label = NULL;
	model->nSV = NULL;

	// read header
	if (!read_model_header(fp, model))
	{
		fprintf(stderr, "ERROR: fscanf failed to read model\n");
		setlocale(LC_ALL, old_locale);
		free(old_locale);
		free(model->rho);
		free(model->label);
		free(model->nSV);
		free(model);
		return NULL;
	}

	// read sv_coef and SV

	int elements = 0;
	long pos = ftell(fp);

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	char *p,*endptr,*idx,*val;

	while(readline(fp)!=NULL)
	{
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
	}
	elements += model->l;

	fseek(fp,pos,SEEK_SET);

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		model->sv_coef[i] = Malloc(double,l);
	model->SV = Malloc(svm_node*,l);
	svm_node *x_space = NULL;
	if(l>0) x_space = Malloc(svm_node,elements);

	int j=0;
	for(i=0;i<l;i++)
	{
		readline(fp);
		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtod(val,&endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
	free(line);

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}

void svm_free_model_content(svm_model* model_ptr)
{
	if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
		free((void *)(model_ptr->SV[0]));
	if(model_ptr->sv_coef)
	{
		for(int i=0;i<model_ptr->nr_class-1;i++)
			free(model_ptr->sv_coef[i]);
	}

	free(model_ptr->SV);
	model_ptr->SV = NULL;

	free(model_ptr->sv_coef);
	model_ptr->sv_coef = NULL;

	free(model_ptr->rho);
	model_ptr->rho = NULL;

	free(model_ptr->label);
	model_ptr->label = NULL;

	free(model_ptr->probA);
	model_ptr->probA = NULL;

	free(model_ptr->probB);
	model_ptr->probB = NULL;

	free(model_ptr->prob_density_marks);
	model_ptr->prob_density_marks = NULL;

	free(model_ptr->sv_indices);
	model_ptr->sv_indices = NULL;

	free(model_ptr->nSV);
	model_ptr->nSV = NULL;
}

void svm_free_and_destroy_model(svm_model** model_ptr_ptr)
{
	if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
	{
		svm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
}

void svm_destroy_param(svm_parameter* param)
{
	free(param->weight_label);
	free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
	// svm_type

	int svm_type = param->svm_type;
	if(svm_type != C_SVC &&
	   svm_type != NU_SVC &&
	   svm_type != ONE_CLASS &&
	   svm_type != EPSILON_SVR &&
	   svm_type != NU_SVR)
		return "unknown svm type";

	// kernel_type, degree

	int kernel_type = param->kernel_type;
	if(kernel_type != LINEAR &&
	   kernel_type != POLY &&
	   kernel_type != RBF &&
	   kernel_type != SIGMOID &&
	   kernel_type != PRECOMPUTED)
		return "unknown kernel type";

	if((kernel_type == POLY || kernel_type == RBF || kernel_type == SIGMOID) &&
	   param->gamma < 0)
		return "gamma < 0";

	if(kernel_type == POLY && param->degree < 0)
		return "degree of polynomial kernel < 0";

	// cache_size,eps,C,nu,p,shrinking

	if(param->cache_size <= 0)
		return "cache_size <= 0";

	if(param->eps <= 0)
		return "eps <= 0";

	if(svm_type == C_SVC ||
	   svm_type == EPSILON_SVR ||
	   svm_type == NU_SVR)
		if(param->C <= 0)
			return "C <= 0";

	if(svm_type == NU_SVC ||
	   svm_type == ONE_CLASS ||
	   svm_type == NU_SVR)
		if(param->nu <= 0 || param->nu > 1)
			return "nu <= 0 or nu > 1";

	if(svm_type == EPSILON_SVR)
		if(param->p < 0)
			return "p < 0";

	if(param->shrinking != 0 &&
	   param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";

	if(param->probability != 0 &&
	   param->probability != 1)
		return "probability != 0 and probability != 1";


	// check whether nu-svc is feasible

	if(svm_type == NU_SVC)
	{
		int l = prob->l;
		int max_nr_class = 16;
		int nr_class = 0;
		int *label = Malloc(int,max_nr_class);
		int *count = Malloc(int,max_nr_class);

		int i;
		for(i=0;i<l;i++)
		{
			int this_label = (int)prob->y[i];
			int j;
			for(j=0;j<nr_class;j++)
				if(this_label == label[j])
				{
					++count[j];
					break;
				}
			if(j == nr_class)
			{
				if(nr_class == max_nr_class)
				{
					max_nr_class *= 2;
					label = (int *)realloc(label,max_nr_class*sizeof(int));
					count = (int *)realloc(count,max_nr_class*sizeof(int));
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}

		for(i=0;i<nr_class;i++)
		{
			int n1 = count[i];
			for(int j=i+1;j<nr_class;j++)
			{
				int n2 = count[j];
				if(param->nu*(n1+n2)/2 > min(n1,n2))
				{
					free(label);
					free(count);
					return "specified nu is infeasible";
				}
			}
		}
		free(label);
		free(count);
	}

	return NULL;
}

int svm_check_probability_model(const svm_model *model)
{
	return
		((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
		 model->probA!=NULL && model->probB!=NULL) ||
		(model->param.svm_type == ONE_CLASS && model->prob_density_marks!=NULL) ||
		((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
		 model->probA!=NULL);
}

void svm_set_print_string_function(void (*print_func)(const char *))
{
	if(print_func == NULL)
		svm_print_string = &print_string_stdout;
	else
		svm_print_string = print_func;
}
