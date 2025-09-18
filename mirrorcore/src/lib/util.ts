/** [0 Bedrock] Generic helpers used across layers. */
export function chunk<T>(arr: T[], size: number){
  const res: T[][] = [];
  for (let i=0;i<arr.length;i+=size){
    res.push(arr.slice(i, i+size));
  }
  return res;
}

export function assertNever(x: never): never {
  throw new Error(`Unexpected value: ${x}`);
}
