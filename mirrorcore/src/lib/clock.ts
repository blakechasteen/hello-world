/** [0 Bedrock] deterministic clock helpers for repeatable tests */
export interface Clock {
  now(): number;
}

export class SystemClock implements Clock {
  now(){
    return Date.now();
  }
}

export class FixedClock implements Clock {
  constructor(private value: number){}
  now(){ return this.value; }
}
