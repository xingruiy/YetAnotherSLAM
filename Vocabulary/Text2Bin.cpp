#include <time.h>
#include "ORBVocabulary.h"

using namespace std;

bool LoadAsText(SLAM::ORBVocabulary *voc, const std::string infile)
{
  clock_t tStart = clock();
  bool res = voc->loadFromTextFile(infile);
  printf("Loading fom text: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
  return res;
}

void LoadAsXML(SLAM::ORBVocabulary *voc, const std::string infile)
{
  clock_t tStart = clock();
  voc->load(infile);
  printf("Loading fom xml: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

void LoadAsBinary(SLAM::ORBVocabulary *voc, const std::string infile)
{
  clock_t tStart = clock();
  voc->loadFromBinaryFile(infile);
  printf("Loading fom binary: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

void SaveAsXML(SLAM::ORBVocabulary *voc, const std::string outfile)
{
  clock_t tStart = clock();
  voc->save(outfile);
  printf("Saving as xml: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

void SaveAsText(SLAM::ORBVocabulary *voc, const std::string outfile)
{
  clock_t tStart = clock();
  voc->saveToTextFile(outfile);
  printf("Saving as text: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

void SaveAsBinary(SLAM::ORBVocabulary *voc, const std::string outfile)
{
  clock_t tStart = clock();
  voc->saveToBinaryFile(outfile);
  printf("Saving as binary: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

int main(int argc, char **argv)
{
  cout << "BoW load/save benchmark" << endl;
  SLAM::ORBVocabulary *voc = new SLAM::ORBVocabulary();

  LoadAsText(voc, "ORBvoc.txt");
  SaveAsBinary(voc, "ORBvoc.bin");

  return 0;
}
