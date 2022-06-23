import { Test, TestingModule } from '@nestjs/testing';
import { SuggestQuestionsService } from './suggest_questions.service';

describe('SuggestQuestionsService', () => {
  let service: SuggestQuestionsService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [SuggestQuestionsService],
    }).compile();

    service = module.get<SuggestQuestionsService>(SuggestQuestionsService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
